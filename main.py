import os
from dotenv import load_dotenv

load_dotenv()

def main():
    print(os.environ.get("PINECONE_API_KEY_DEFAULT"))
    print("Hello from rag-intro!")


if __name__ == "__main__":
    main()

