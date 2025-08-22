from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever

def create_retriever(
    subfolder: str,
    top_k: int = 10,
) -> VectorStoreRetriever:
    """
    Load a local FAISS vector store and return the retriever.
    """
    # choose embedding model
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    # load the FAISS store from disk
    vector_store = FAISS.load_local(
        subfolder,
        embedding,
        allow_dangerous_deserialization=True,  # required when loading from local disk
    )

    # initialize the retriever
    return vector_store.as_retriever(search_kwargs={"k": top_k})
