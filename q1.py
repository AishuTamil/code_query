import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from sklearn.preprocessing import normalize


faiss_index = faiss.read_index('faiss_db')
metadata = np.load('metadata.npy', allow_pickle=True)
chunks = np.load('chunks.npy', allow_pickle=True)


embeddings = OpenAIEmbeddings()

def process_user_query(user_query):
    user_query_embedding = embeddings.embed_query(user_query)
    
    user_query_embedding = normalize([user_query_embedding])
    
    k = 5  
    distances, indices = faiss_index.search(user_query_embedding, k)
    relevant_chunks = [chunks[idx] for idx in indices[0]]
    relevant_metadata = [metadata[idx] for idx in indices[0]]
    print("Relevant:",relevant_metadata)
    context = "\n".join(relevant_chunks)
    
    prompt_template = """
    You are an AI assistant. Based on the following context, answer the user's question concisely and clearly:
    
    Context:
    {context}
    
    User Question:
    {question}
    
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    prompt_with_context = prompt.format(context=context, question=user_query)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    response = llm.run(prompt_with_context)
    
    return response

if __name__ == "__main__":
 
    user_query = ""

   
    response = process_user_query(user_query)
    print("Response:", response)
