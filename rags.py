# Install necessary libraries before running this code:
# pip install langchain langchain-community langchain-openai langchain-chroma langchain-text-splitters chromadb python-dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma # Assuming Chroma is the vector store being used
from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Load the document
loader = TextLoader("ai_intro.txt")
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings()

# Create a vector store from the text chunks
vectorstore = Chroma.from_documents(texts, embeddings)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever()

# Example query
query = "What is LangChain used for?"

# Retrieve relevant documents
docs = retriever.invoke(query)

# Combine the documents into context
context = "\n\n".join([doc.page_content for doc in docs])

# Create a prompt with context
prompt = f"""Use the following context to answer the question:

Context: {context}

Question: {query}

Answer:"""

# Get answer from LLM
llm = OpenAI()
answer = llm.invoke(prompt)
print(answer) # Note: Make sure to set your OpenAI API key in the environment variables before running this code