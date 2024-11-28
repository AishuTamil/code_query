import os
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
mongo_uri = os.getenv("mongodb cluster") 
client = MongoClient(mongo_uri)
db = client['sample']  
collection = db['check']  
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
def cosine_distance(a, b):
    return 1 - sum([a_i * b_i for a_i, b_i in zip(a, b)]) / (
            sum([a_i ** 2 for a_i in a]) ** 0.5 * sum([b_i ** 2 for b_i in b]) ** 0.5)

def nearest_embedding(user_embedding, collection):
    item_distances = []
    for item in collection.find({}, {"embedding": 1, "text": 1, "pdf_name": 1}):
        emb_i = item["embedding"]
        distance = cosine_distance(user_embedding, emb_i)
        formatted_output = item['text']
        item_distances.append((distance, formatted_output, item['pdf_name']))
    
    item_distances.sort(key=lambda x: x[0])
    top_2_items = item_distances[:2]
    
    return top_2_items
def generate_response_from_llm(similar_content, user_query):
    context = "\n".join([f"Content: {item[1]}" for item in similar_content])
    qa_system_prompt = f"""You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", user_query),
        ]
    )
    
    response = llm(qa_prompt.format_messages(input=user_query))
    return response[0]['text']
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    user_embedding = embedding_model.encode(user_query).tolist()
    similar_content = nearest_embedding(user_embedding, collection)
    response = generate_response_from_llm(similar_content, user_query)
    print("Response:", response)

