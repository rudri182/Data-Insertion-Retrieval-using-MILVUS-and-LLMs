import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import numpy as np
import json
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel
from gensim import corpora, models
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")
utility.drop_collection("embedding_ivf_new")


nltk.download('punkt')
nltk.download('stopwords')

visited_urls = set()
stop_words = set(stopwords.words('english'))

# To load the scrapped data from directory
def load_text_data(directory):
    texts = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                texts.append(f.read())
    return texts

# To tokenize the text into sentences and remove stopwords
def preprocess_text(text):
    sentences = sent_tokenize(text)
    processed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        processed_sentences.append(' '.join(filtered_words))
    return processed_sentences

# To generate embeddings for each sentence
def get_bert_embeddings(sentences):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())
    return embeddings

# To cluster sentences into chunks based on similarity
def chunk_sentences(sentences, embeddings, num_chunks):
    kmeans = KMeans(n_clusters=num_chunks)
    kmeans.fit(embeddings)
    clusters = kmeans.predict(embeddings)
    chunks = [[] for _ in range(num_chunks)]
    for i, cluster in enumerate(clusters):
        chunks[cluster].append(sentences[i])
    return chunks

# To ensure chunks are topic-relevant using LDA
def topic_modeling(chunks):
    dictionary = corpora.Dictionary(chunks)
    corpus = [dictionary.doc2bow(chunk) for chunk in chunks]
    lda_model = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)
    topics = lda_model.print_topics(num_words=4)
    for topic in topics:
        print(topic)


def save_embeddings_to_milvus(embeddings, documents, urls, collection_name, index_params):
    connections.connect("default", host="localhost", port="19530")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500)
    ]
    schema = CollectionSchema(fields, description="Vector database for embeddings with metadata")

    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
    else:
        collection = Collection(name=collection_name)

    data = [
        # [i for i in range(len(embeddings))],  # IDs
        embeddings,                  # Embedding vectors
        # documents,                            # Text documents
        urls                                  # Metadata URLs
    ]
    collection.insert(data)
    collection.release()  # Release the collection before creating the index
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()





# def execute(directory, num_chunks=5):
#     texts = load_text_data(directory)
#     all_sentences = []
#     for text in texts:
#         all_sentences.extend(preprocess_text(text))
    
#     embeddings = get_bert_embeddings(all_sentences)
#     chunks = chunk_sentences(all_sentences, embeddings, num_chunks)
#     chunk_embeddings = get_bert_embeddings([" ".join(chunk) for chunk in chunks])
#     topic_modeling(chunks)
    
#     metadata = [f"chunk_{i}_url" for i in range(len(chunk_embeddings))]

    
#     # Save with IVF index
#     save_embeddings_to_milvus(chunk_embeddings, metadata, "embedding_ivf", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})

#     return chunk_embeddings

def execute(directory, num_chunks=5):
    texts = load_text_data(directory)
    all_sentences = []
    for text in texts:
        all_sentences.extend(preprocess_text(text))
    
    embeddings = get_bert_embeddings(all_sentences)
    chunks = chunk_sentences(all_sentences, embeddings, num_chunks)
    chunk_embeddings = get_bert_embeddings([" ".join(chunk) for chunk in chunks])
    topic_modeling(chunks)

    print("Chunks:--------- ", chunks)
    
    documents = [" ".join(chunk) for chunk in chunks]
    urls = [f"chunk_{i}_url" for i in range(len(chunk_embeddings))]

    # # Save with FLAT index
    # save_embeddings_to_milvus(chunk_embeddings, documents, urls, "embedding_flat", {"index_type": "FLAT", "metric_type": "L2", "params": {}})
    
    # Save with IVF index
    save_embeddings_to_milvus(chunk_embeddings, documents, urls, "embedding_ivf_new", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})

    return chunk_embeddings




for i in range(5):

    directory = 'level_' + str(i)
    chunk_embeddings = execute(directory)

