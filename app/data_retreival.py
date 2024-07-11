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

collection_name_ivf = "embedding_ivf"

# Check if collection exists
if not Collection.exists(collection_name_ivf):
    collection_ivf = Collection(name=collection_name_ivf)

# Function to search using BM25
def bm25_search(query, documents, k=10):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_docs = [tokenizer.tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = tokenizer.tokenize(query)
    doc_scores = bm25.get_scores(query_tokens)
    sorted_indices = np.argsort(doc_scores)[::-1]
    return sorted_indices[:k]

# Function to get DPR embeddings
def get_dpr_embeddings(texts):
    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        context_embeddings = context_encoder(**inputs)
    return context_embeddings.last_hidden_state.mean(dim=1).numpy()

# Search in Milvus
def search_in_milvus(collection_name, query_embedding, top_k=10):
    collection = Collection(collection_name)
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    results = collection.search(query_embedding, "embedding", search_params, limit=top_k, output_fields=["url"])
    return results

# Query expansion using WordNet
def expand_query(query):
    synonyms = set()
    for word in nltk.word_tokenize(query):
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    return list(synonyms)

# Hybrid Retrieval Function
def hybrid_retrieve(query, top_k=10):
    
    # Query expansion
    expanded_query = " ".join(expand_query(query))
    
    # BM25 Search
    bm25_docs = [doc['text'] for doc in metadata]
    bm25_indices = bm25_search(expanded_query, bm25_docs, k=top_k)
    bm25_results = [metadata[i] for i in bm25_indices]

    # DPR Embedding Search
    query_tokens = BertTokenizer.from_pretrained('bert-base-uncased').encode(expanded_query, return_tensors="pt")
    with torch.no_grad():
        query_embedding = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')(
            input_ids=query_tokens
        ).last_hidden_state.mean(dim=1).numpy()
    
    milvus_results = search_in_milvus(collection_name_ivf, query_embedding, top_k)
    milvus_results = [doc for doc in milvus_results if doc['url'] not in [result['url'] for result in bm25_results]]
    
    # Combine results and re-rank
    combined_results = bm25_results + milvus_results
    combined_embeddings = [embedding for result in combined_results for embedding in result['embedding']]
    similarities = cosine_similarity(query_embedding, combined_embeddings)[0]
    
    for i, result in enumerate(combined_results):
        result['similarity'] = similarities[i]
    
    combined_results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return combined_results[:top_k]

# Question Answering Function
def answer_question(question, context):
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")
    
    answer = qa_pipeline(question=question, context=context)
    return answer['answer']


# Example Query
query = "What are the features of CUDA?"
results = hybrid_retrieve(query, top_k=10)

# Print results
for i, result in enumerate(results):
    print(f"Rank {i+1}: {result['url']} - Similarity Score: {result['similarity']}")

# Prepare context for question answering
context = " ".join([result['text'] for result in results])

# Get the answer from the question answering model
answer = answer_question(query, context)
print(f"Question: {query}")
print(f"Answer: {answer}")