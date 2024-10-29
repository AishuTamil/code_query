import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import transformers
import torch
# Initialize the Sentence Transformer model for creating embeddings
os.environ['HF_TOKEN'] = "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"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

model_id = "google/flan-t5-large"
model = SentenceTransformer('all-mpnet-base-v2')
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Load the saved FAISS index
faiss_index = faiss.read_index('faiss_db')

# Load the metadata and chunks
metadata = np.load('metadata.npy', allow_pickle=True).tolist()
chunks = np.load('chunks.npy', allow_pickle=True).tolist()

# Function to query the vector store and get the most relevant chunks
def query_vector_store(query, top_k=3):
    # Convert the query to a vector
    query_embedding = model.encode([query])
    
    # Search for the most similar document chunks
    distances, indices = faiss_index.search(np.array(query_embedding), top_k)
    
    # Retrieve the closest chunks with metadata
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        result = {
            'filename': metadata[idx]['filename'],
            'chunk_number': metadata[idx]['chunk'],
            'distance': dist,
            'content': chunks[idx]  
        }
        results.append(result)
    
    return results
    
def engage_with_llm(query, results):
 
    system_prompt = (
        "You are an AI assistant specialized in sustainability reporting standards. "
         "Transform the content into a brief, new response for the user's question."
    )
    relevant_content = "\n".join(result['content'] for result in results)
    user_prompt = f"Utilize {relevant_content} and answer concisely for the user queries. **User Query:** {query}"
                  
                  
    chat_context = f"{system_prompt}\n\n{user_prompt}"
    # print("Chat Context:")
    # print(chat_context)
    generation_pipeline = transformers.pipeline(
        "text2text-generation", model=llm_model, tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto", trust_remote_code=True
    )
    output = generation_pipeline(chat_context, max_new_tokens=150,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,)
    print("Output")
    return output[0]['generated_text']
   

# Example query
query = "What is CDL"
results = query_vector_store(query, top_k=3)
response = engage_with_llm(query, results)
# for result in results:
#     print("checking")
#     print(f"Relevant Content: {result['content']}")
#     print("-" * 80)

print("Response:")
print(response)
