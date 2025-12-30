import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def main():
    print(os.environ.get("PINECONE_API_KEY_DEFAULT"))
    print("Hello from rag-intro!")


if __name__ == "__main__":
    main()

