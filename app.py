import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
load_dotenv()
#Sample text representing knowledge base
sample_knowledge_base = """The company remote work policy allows employees to work from anywhere in the world for up to 90 days per calendar year. 
However, employees must maintain core working hours between 10:00 AM and 3:00 PM EST. 
Expense reimbursement for home office setups is capped at $500 per employee, refreshable every two years."""
print("--- Raw Document Loaded ---")
print(f"Total Characters: {len(sample_knowledge_base)}")
#Splitting the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 150, 
    chunk_overlap = 20
)
#Running the splitter
text_chunks = text_splitter.create_documents([sample_knowledge_base])
print("--- Document Successfully Chunked ---")
for i,chunk in enumerate(text_chunks):
    print(f"Chunk Number {(i+1)}:")
    print(chunk.page_content)
print("--- OpenAI Embeddings and ChromaDB ---")
#Creating the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
#Initialising ChromaDB and saving text chunks directly into my Mac
vector_store = Chroma.from_documents(
    documents = text_chunks,
    embedding = embedding_model,
    persist_directory = "./chroma_db"
)
#Defining a test question
user_question = "When is my next reimbursement to be enhanced ?"
print(f"User Question : {user_question}")
#Searching the database for the 2 closest matching text chunks
matching_chunks = vector_store.similarity_search(user_question,k = 2)
print("Top Matching Chunks found in the database:")
for i,chunk in enumerate(matching_chunks):
    print(f"\nMatch Number {i+1}:")
    print(chunk.page_content)
print("--- Generating an LLM response ---")
#Joining retrived text chunks into a single text block
context_text = "/n/n".join([chunk.page_content for chunk in matching_chunks])
#Designing a strict prompt forcing the model to act as a grounded assistant
system_prompt = f"""
You are a helpful company HR assistant.
Answer the user's question using ONLY the provided text context below.
If the answer cannot be found in the context, say "I cannot find that information in the document."

Context:
{context_text}
"""
#Initialising OpenAI's chat model and setting the temperature to 0 so that the model acts deterministically and doesnt hallucinate
llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
#Invoking the model by passing the system prompt and user question
ai_response = llm.invoke([
    ("system", system_prompt),
    ("human", user_question)
]) 
print("\n--- Final ChatBot Answer ---")
print(ai_response.content)   

