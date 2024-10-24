import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM 
import faiss
# Initialize the Sentence Transformer model for creating embeddings
model = SentenceTransformer('all-mpnet-base-v2',cache_folder=None)
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b", cache_dir=None)
llm_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b",cache_dir=None)
os.environ["HF_HOME"] = "D:\New folder (2)"

# Load the saved FAISS index
faiss_index = faiss.read_index('faiss_db')

# Load the metadata and chunks
metadata = np.load('metadata.npy', allow_pickle=True).tolist()
chunks = np.load('chunks.npy', allow_pickle=True).tolist()

# Function to query the vector store and get the most relevant chunks
def query_vector_store(query, top_k=5):
    # Convert the query to a vector
    query_embedding = model.encode([query])
    
    # Search for the most similar document chunks
    distances, indices = faiss_index.search(np.array(query_embedding), top_k)
    
    # Retrieve the closest chunks with metadata
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        result = {

            'content': chunks[idx]  # Show first 200 characters of the relevant chunk
        }
        results.append(result)
    
    return results
def engage_with_llm(query, results):
    
    chat_context = [
        "You are a genAI expert. Engage in a conversation with the user based on the relevant content provided below.",
        f"User query: {query}"
    ]

    relevant_contents = [f"Relevant Content: {result['content']}" for result in results]
    chat_context += relevant_contents

    chat_context_str = "\n".join(chat_context)

    input_ids = tokenizer(chat_context_str, return_tensors="pt", padding=True, truncation=True).input_ids
    output = llm_model.generate(input_ids, max_length=5000, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)
# Example query
query = "Channels to reach these audience"
results = query_vector_store(query, top_k=10)
response = engage_with_llm(query, results)
# Print the results
for result in results:
    print(f"Document: {result['filename']}, Chunk: {result['chunk_number']}, Distance: {result['distance']}")
    print(f"Relevant Content: {result['content']}")
    print("-" * 80)
print("Response:")
print(response)
