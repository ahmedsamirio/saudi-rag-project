from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
import uuid


def get_parent_child_splits(docs, parent_chunk_size=1200, parent_chunk_overlap=400, child_chunk_size=300, child_chunk_overlap=0, id_key="parent_doc_id"):
    parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=['\n\n\n', '\n\n', '\n', r'\.\s+', ' ', '']
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )

    parent_docs = parent_splitter.split_documents(docs)
    parent_docs_ids = [str(uuid.uuid4()) for _ in parent_docs]

    child_docs = []
    for i, doc in enumerate(parent_docs):
        _id = parent_docs_ids[i]
        _child_docs = child_splitter.split_documents([doc])
        for _doc in _child_docs:
            _doc.metadata[id_key] = _id
        child_docs.extend(_child_docs)

    return parent_docs, parent_docs_ids, child_docs


def get_embedding_function(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)


def get_multivector_retriever(chroma_client, embedding_model_name, collection_name, save_path, parent_docs=[], parent_docs_ids=[], child_docs=[], id_key="parent_doc_id"):
    
    # Create save directories
    os.makedirs(os.path.join(save_path), exist_ok=True)
    docstore_path = os.path.join(save_path, 'docstore', collection_name)
    vectorstore_path = os.path.join(save_path, 'chroma')

    # Get embedding_function
    embedding_function = get_embedding_function(embedding_model_name)

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=vectorstore_path
    )

    store = LocalFileStore(docstore_path)

    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )

    # If vectorstore isn't populated, populate and persist
    if not len(retriever.vectorstore.get()['documents']) == len(child_docs):

        if child_docs:
            retriever.vectorstore.add_documents(child_docs)
            retriever.docstore.mset(list(zip(parent_docs_ids, parent_docs)))

            # Save the vectorstore and docstore to disk
            retriever.vectorstore.persist()

    return retriever      