import nltk
import numpy as np
import json
import torch

from sklearn.metrics.pairwise import cosine_similarity
from transformers import DPRContextEncoder, DPRQuestionEncoder, BertTokenizer, pipeline
from rank_bm25 import BM25Okapi
from pymilvus import connections, Collection
from nltk.corpus import wordnet


# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Load embeddings and metadata
def load_embeddings(json_path, npy_path):
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    embeddings = np.load(npy_path)
    return embeddings, metadata

embeddings, metadata = load_embeddings('embeddings.json', 'embeddings.npy')

collection_name_ivf = "embedding_ivf_new"

# Function to search using BM25
def bm25_search(query, documents, k=10):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_docs = [tokenizer.tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = nltk.word_tokenize(str(query).lower())  # Ensure query is a string
    doc_scores = bm25.get_scores(query_tokens)
    sorted_indices = np.argsort(doc_scores)[::-1]
    return sorted_indices[:k]
    
# Query expansion using WordNet
def expand_query(query):
    synonyms = set()
    for word in nltk.word_tokenize(query):
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    return list(synonyms)

# Question Answering Function
def answer_question(question, context):
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")
    
    answer = qa_pipeline(question=question, context=context)
    return answer['answer']

# Function to search in Milvus
def search_in_milvus(collection_name, query_embedding, top_k=10):
    collection = Collection(name=collection_name)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(query_embedding, "embedding", search_params, limit=top_k)
    return results

# Hybrid Retrieval Function
def hybrid_retrieve(query, top_k=10):
    # Step 1: Query Expansion
    expanded_query = expand_query(query)
    
    # Step 2: BM25 Search
    # Converting vectors to strings
    bm25_docs = [" ".join([str(num) for num in vec]) for vec in embeddings]  
    bm25_indices = bm25_search(expanded_query, bm25_docs, k=top_k)
    bm25_results = [embeddings[i] for i in bm25_indices]

    # Step 3: DPR Embedding Search
    tokenizer = BertTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    query_inputs = tokenizer(expanded_query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        query_embedding = question_encoder(**query_inputs).pooler_output.numpy()
    
    milvus_results = search_in_milvus(collection_name_ivf, query_embedding, top_k)
    
    # Step 4: Combine Results
    combined_results = bm25_results + [result for result in milvus_results[0]]
    
    # Step 5: Re-rank Combined Results  
    combined_embeddings = combined_results
    
    combined_embeddings = combined_embeddings[0].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, combined_embeddings)
    
    ranked_results = sorted(zip(combined_results, similarities), key=lambda x: x[1], reverse=True)
    
    return ranked_results[:top_k]

# Example Query
query = "What are the features of CUDA?"
results = hybrid_retrieve(query, top_k=10)

# Print results
for i, (embedding, similarity) in enumerate(results):
    print(f"Rank {i+1}: Similarity Score: {similarity}")

# Prepare context for question answering
context = " ".join([str(result) for result in results])

# Get the answer from the question answering model
answer = answer_question(query, context)
print(f"Question: {query}")
print(f"Answer: {answer}")
