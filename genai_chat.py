import os
import json
import fitz  # PyMuPDF
import traceback
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Set your Hugging Face token here
os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_token_here"  

# MongoDB configuration
MONGO_URI = "name_cluster"
DB_NAME = "sample_data"
COLLECTION_NAME = "check_data"

# Initialize models and clients
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight sentence transformer for embeddings
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Initialize DialoGPT model for response generation
# dialogpt_model = pipeline("text-generation", model="microsoft/DialoGPT-medium")

def extract_text_from_pdf(pdf_path):
    text_content = []
    with fitz.open(pdf_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text_content.append(page.get_text("text"))
    return text_content

def store_embeddings_in_db(texts, pdf_name):
    for text in texts:
        embedding = embedding_model.encode(text).tolist()
        collection.insert_one({
            "pdf_name": pdf_name,
            "text": text,
            "embedding": embedding
        })
    print(f"Embeddings from {pdf_name} successfully stored in MongoDB.")

# def cosine_similarity(vec_a, vec_b):
#     return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

# def retrieve_similar_text(query):
#     query_embedding = embedding_model.encode(query)
#     results = collection.find({})
#     similarities = []

#     for item in results:
#         stored_embedding = np.array(item["embedding"])
#         similarity = cosine_similarity(query_embedding, stored_embedding)
#         similarities.append((similarity, item["text"], item["pdf_name"]))

#     # Sort by similarity and return the best match
#     if similarities:
#         similarities.sort(reverse=True, key=lambda x: x[0])
#         best_match_text, pdf_name = similarities[0][1], similarities[0][2]
#         print(f"Best match text from '{pdf_name}': {best_match_text}")
#         return best_match_text
#     else:
#         print("No similar text found.")
#         return None



def process_pdfs_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            print(f"Processing PDF: {filename}")
            pdf_texts = extract_text_from_pdf(pdf_path)
            store_embeddings_in_db(pdf_texts, filename)

# Main Execution
if __name__ == "__main__":
    try:
        pdf_directory = r"C:\Users\12400\Downloads\LLM-main (2)\LLM-main\Emp\Documents"
        process_pdfs_in_directory(pdf_directory)

        # user_query = input("Enter your query: ")
        # similar_text = retrieve_similar_text(user_query)
        

    except Exception as e:
        print("Error:", traceback.format_exc())
