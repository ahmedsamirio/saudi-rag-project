from config import DATA_PATH, CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME, MODEL_NAME
from indexing import get_multivector_retriever
from generation import ROUTER_SYSTEM_PROMPT, ROUTER_PROMPT, QA_SYSTEM_PROMPT, QA_PROMPT, SUMMARY_SYSTEM_PROMPT, SUMMARY_PROMPT, FINAL_SUMMARY_SYSTEM_PROMPT, FINAL_SUMMARY_PROMPT, LLAMA_PROMPT_TEMPLATE, MIXTRAL_PROMPT_TEMPLATE
from generation import get_model, format_docs, get_rag_chain, get_router_chain, summarize

from langchain_core.prompts import PromptTemplate

import chromadb

import gradio as gr


if __name__ == "__main__":
    persistent_client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    if "mixtral" in MODEL_NAME:
        MODEL_TEMPLATE = MIXTRAL_PROMPT_TEMPLATE

    elif "llama" in MODEL_NAME:
        MODEL_TEMPLATE = LLAMA_PROMPT_TEMPLATE

    router_prompt_template = PromptTemplate.from_template(MODEL_TEMPLATE.format(system_prompt=ROUTER_SYSTEM_PROMPT, user_message=ROUTER_PROMPT))
    qa_prompt_template = PromptTemplate.from_template(MODEL_TEMPLATE.format(system_prompt=QA_SYSTEM_PROMPT, user_message=QA_PROMPT))
    summarization_prompt_template = PromptTemplate.from_template(MODEL_TEMPLATE.format(system_prompt=SUMMARY_SYSTEM_PROMPT, user_message=SUMMARY_PROMPT))
    final_summary_template = PromptTemplate.from_template(MODEL_TEMPLATE.format(system_prompt=FINAL_SUMMARY_SYSTEM_PROMPT, user_message=FINAL_SUMMARY_PROMPT))

    llm = get_model(MODEL_NAME)
    qa_retriever = get_multivector_retriever(persistent_client, EMBEDDING_MODEL_NAME, COLLECTION_NAME, DATA_PATH)
    summary_retriever = get_multivector_retriever(persistent_client, EMBEDDING_MODEL_NAME, COLLECTION_NAME, DATA_PATH)

    router_chain = get_router_chain(llm, router_prompt_template)
    rag_chain = get_rag_chain(llm, retriever, format_docs, qa_prompt_template)
    summarization_chain = get_rag_chain(llm, retriever, format_docs, summarization_prompt_template)

    router_chain = 

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
