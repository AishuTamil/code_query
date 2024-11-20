import os
import PyPDF2
from langchain_community.embeddings import OpenAIEmbeddings
import chromadb
client = chromadb.Client()
# os.environ["OPENAPI_KEY"]=""
collection_name = "pdf_embeddings"
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
def v_store(page_texts):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    page_embeddings = [embeddings.embed_documents([page_text])for page_text in page_texts]
    page_embeddings = [embedding[0] for embedding in page_embeddings]
    return page_embeddings, page_texts 
def store_chroma(embeddings, texts):
    collection = client.create_collection(collection_name)
    ids = [str(i) for i in range(len(embeddings))] 
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts  
    )
    print("Embeddings and corresponding texts stored in Chroma.")
    
if __name__ == "__main__":
    f_path = r"C:\Users\12400\Downloads\LLM-main (2)\LLM-main\Emp\Documents"
    pdf_files = folder(f_path) 
    for pdf_file in pdf_files:
        extracted_text = extract_content(pdf_file)
        print(f"Extracted Text from {pdf_file}:\n{extracted_text}...") 
        data = v_store(extracted_text)
        print(data)
        embeddings, texts = data
        st=store_chroma(embeddings, texts)
        print(st)
