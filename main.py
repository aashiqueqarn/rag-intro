import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
print("*" * 100)
print("Initialing Components")
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-5.2")
vector_store = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"), embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    Provide a detailed answer:"""
)
print("Components Initialized")
print("*" * 100)

def format_docs(docs):
    """Formats the retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str) -> str:
    """
    Simple retrieval chain without LCEl.
    Manually retrieves documents, formats them and generate a response

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step1: Retrieve relevant documents
    docs = retriever.invoke(query)
    # Step 2: Format documents into context string
    context = format_docs(docs)
    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)
    # Step 4: Invoke the LLM with the formatted messages
    response = llm.invoke(messages)
    # Step 5: Return the content
    return response.content



if __name__ == "__main__":
    print("Retrieving ...")

    user_query = "What is Pinecone in machine learning?"
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=user_query)])
    print("\n Answer:")
    print(result_raw.content)

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Implementation without LCEl")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(user_query)
    print("\n Answer:")
    print(result_without_lcel)
