import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline
mongo_uri = os.getenv('MONGO_URI',"mongo url") 
client = MongoClient(mongo_uri)
db = client['sample_data']  
collection = db['check_data']  
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embedding_model = SentenceTransformer(embedding_model_name)

llm_model_name = 'microsoft/DialoGPT-medium'
llm_pipeline = pipeline('text-generation', model=llm_model_name)

# Initialize the summarization model
summarization_model_name = 'facebook/bart-large-cnn'
summarization_pipeline = pipeline('summarization', model=summarization_model_name)

def cosine_distance(a, b):
    return 1 - sum([a_i * b_i for a_i, b_i in zip(a, b)]) / (
            sum([a_i ** 2 for a_i in a]) ** 0.5 * sum([b_i ** 2 for b_i in b]) ** 0.5)

def nearest_embedding_ocr(user_embedding, collection):
    item_distances = []
    for item in collection.find({}, {"embedding": 1, "text": 1, "pdf_name": 1}):
        emb_i = item["embedding"]
        distance = cosine_distance(user_embedding, emb_i)
        formatted_output = item['text']
        item_distances.append((distance, formatted_output, item['pdf_name']))
    
    item_distances.sort(key=lambda x: x[0])
    top_2_items = item_distances[:2]
    
    return top_2_items

def summarize_content(contents):
    """Summarize the contents."""
    summaries = []
    max_input_length = 1024  
    for content in contents:
        content_text = content[1][:max_input_length]  
        if isinstance(content_text, str):
            summary = summarization_pipeline(content_text, max_length=50, min_length=25, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        else:
            print("Content is not a string:", content_text)
    return summaries


def generate_response_from_llm(summarized_content):
    prompt = " ".join(summarized_content)
    response = llm_pipeline(prompt, max_new_tokens=100, num_return_sequences=1)[0]['generated_text']
    return response

def main():
    user_query = input("Enter your query: ")
    user_embedding = embedding_model.encode(user_query)
    similar_content = nearest_embedding_ocr(user_embedding, collection)

    if similar_content:
        summarized_content = summarize_content(similar_content)

        response = generate_response_from_llm(summarized_content)
        print("Response:", response)
    else:
        print("No similar content found.")

if __name__ == "__main__":
    main()
