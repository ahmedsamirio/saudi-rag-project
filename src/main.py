from config import DATA_PATH, CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME, MODEL_NAME
from indexing import get_multivector_retriever
from generation import QA_SYSTEM_PROMPT, QA_PROMPT, LLAMA_PROMPT_TEMPLATE, MIXTRAL_PROMPT_TEMPLATE
from generation import get_model, format_docs, get_rag_chain

from langchain_core.prompts import PromptTemplate

import chromadb

import gradio as gr


if __name__ == "__main__":
    persistent_client = chromadb.PersistentClient(path=CHROMA_PATH)
    retriever = get_multivector_retriever(persistent_client, EMBEDDING_MODEL_NAME, COLLECTION_NAME, DATA_PATH)

    if "mixtral" in MODEL_NAME:
        qa_prompt_template = PromptTemplate.from_template(MIXTRAL_PROMPT_TEMPLATE.format(system_prompt=QA_SYSTEM_PROMPT, user_message=QA_PROMPT))

    elif "llama" in MODEL_NAME:
        qa_prompt_template = PromptTemplate.from_template(LLAMA_PROMPT_TEMPLATE.format(system_prompt=QA_SYSTEM_PROMPT, user_message=QA_PROMPT))

    llm = get_model(MODEL_NAME)
    rag_chain = get_rag_chain(llm, retriever, format_docs, qa_prompt_template)

    def get_answer(question, history):
        return rag_chain.invoke(question)

    gr.ChatInterface(
        get_answer,
        chatbot=gr.Chatbot(height=500, rtl=True),
        textbox=gr.Textbox(placeholder="Ask me a question", container=False, scale=7, rtl=True),
        title="Arabic RAG Chatbot",
        description="",
        theme="soft",
        # examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
        # cache_examples=True,
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear",
    ).launch()
