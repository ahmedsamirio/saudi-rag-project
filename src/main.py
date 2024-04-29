from config import PROD_STORAGE_PATH, EMBEDDING_MODEL_NAME, MODEL_NAME, QA_COLLECTION_NAME, SUMMARY_COLLECTION_NAME
from indexing import get_multivector_retriever
from generation import ROUTER_SYSTEM_PROMPT, ROUTER_PROMPT, QA_SYSTEM_PROMPT, QA_PROMPT, SUMMARY_SYSTEM_PROMPT, SUMMARY_PROMPT, FINAL_SUMMARY_SYSTEM_PROMPT, FINAL_SUMMARY_PROMPT, LLAMA_PROMPT_TEMPLATE, MIXTRAL_PROMPT_TEMPLATE
from generation import get_model, format_docs, get_rag_chain, get_router_chain, summarize

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


import chromadb

import gradio as gr
import os

import argparse


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="Load different models for a retrieval augmented generation.")
    parser.add_argument("--model", default="mixtral-8x22b", choices=["mixtral-8x22b", "llama-3-8b", "llama-3-70b"], help="Select the model to load")
    parser.add_argument("--tree-summarizer", action='store_true', help="Enable tree summarization in summary questions")
    args = parser.parse_args()

    # Model mapping
    model_map = {
        "mixtral-8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "llama-3-8b": "meta-llama/Llama-3-8b-chat-hf",
        "llama-3-70b":  "meta-llama/Llama-3-70b-chat-hf"
    }

    MODEL_NAME = model_map[args.model]
    
    persistent_client = chromadb.PersistentClient(path=os.path.join(PROD_STORAGE_PATH, "chroma"))
    
    if "mistral" in MODEL_NAME:
        MODEL_TEMPLATE = MIXTRAL_PROMPT_TEMPLATE

    elif "llama" in MODEL_NAME:
        MODEL_TEMPLATE = LLAMA_PROMPT_TEMPLATE

    router_prompt_template = PromptTemplate.from_template(MODEL_TEMPLATE.format(system_prompt=ROUTER_SYSTEM_PROMPT, user_message=ROUTER_PROMPT))
    qa_prompt_template = PromptTemplate.from_template(MODEL_TEMPLATE.format(system_prompt=QA_SYSTEM_PROMPT, user_message=QA_PROMPT))
    summarization_prompt_template = PromptTemplate.from_template(MODEL_TEMPLATE.format(system_prompt=SUMMARY_SYSTEM_PROMPT, user_message=SUMMARY_PROMPT))
    final_summary_template = PromptTemplate.from_template(MODEL_TEMPLATE.format(system_prompt=FINAL_SUMMARY_SYSTEM_PROMPT, user_message=FINAL_SUMMARY_PROMPT))

    print("Loading", MODEL_NAME)
    llm = get_model(MODEL_NAME)

    print("Loading Retrievers")
    qa_retriever = get_multivector_retriever(persistent_client, EMBEDDING_MODEL_NAME, QA_COLLECTION_NAME, PROD_STORAGE_PATH, k=15)
    summary_retriever = get_multivector_retriever(persistent_client, EMBEDDING_MODEL_NAME, SUMMARY_COLLECTION_NAME, PROD_STORAGE_PATH, k=15)


    router_chain = get_router_chain(llm, router_prompt_template)
    rag_chain = get_rag_chain(llm, qa_retriever, format_docs, qa_prompt_template)
    summarization_chain = get_rag_chain(llm, summary_retriever, format_docs, summarization_prompt_template)
    final_answer_chain = (
        final_summary_template
        | llm
        | StrOutputParser()
    )

    def get_answer(question, history):
        
        routing = router_chain.invoke(question)
        print(routing)
        if "FACT" in routing.upper():
            docs = qa_retriever.invoke(question)
            answer = rag_chain.invoke(question)
        else:

            if args.tree_summarizer:
                retrieved_docs = summary_retriever.invoke(question)
                summary_answers = summarization_chain.batch({"question": [question]*len(retrieved_docs), "context": [d.page_content for d in retrieved_docs]})
                answer = final_answer_chain.invoke({"question": question, "context": "\n\n".join(summary_answers)})

            else:
                answer = summarization_chain.invoke(question)

        return answer

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