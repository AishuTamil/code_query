import os
import PyPDF2
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
# os.environ["OPENAPI_KEY"]=""
MONGO_URI = "mongodb cluster"
DB_NAME = "sample"
COLLECTION_NAME = "check"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def folder(folder_path):
    pdfs = []
    for p in os.listdir(folder_path):
        path = os.path.join(folder_path, p)
        print(path)
        if p.endswith(".pdf"):
            pdfs.append(path)  
    return pdfs

def extract_content(pdf_path):
    pages_content = [] 
    with open(pdf_path, 'rb') as file:  
        pdf_reader = PyPDF2.PdfReader(file)  
        for page_num in range(len(pdf_reader.pages)):  
            page = pdf_reader.pages[page_num]  
            text = page.extract_text()  
            if text:
                cleaned_text = " ".join(text.split())
                pages_content.append(cleaned_text) 
    return pages_content
def store_embeddings_in_db(pages, pdf_name):
    try:
        for page in pages:
            text = page["text"]
            embedding = embedding_model.encode(text).tolist()
            collection.insert_one({
                "pdf_name": pdf_name,
                "text": text,
                "embedding": embedding
            })
        print(f"Embeddings from {pdf_name} successfully stored in MongoDB.")
    except Exception as e:
        print(f"Error storing embeddings for {pdf_name}")
if __name__=="__main__":
    path="C:/Users/12400/Downloads/Documents"
    pdf_files=folder(path)
    for pdf in pdf_files:
        extracted_text=extract_content(pdf)
        pdf_name = os.path.basename(pdf)
        store_embeddings_in_db(extracted_text, pdf_name)
        store_data=store_embeddings_in_db(extracted_text,pdf_name)
        print(store_data)

        