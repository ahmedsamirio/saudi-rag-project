from langchain_together import Together
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from pathlib import Path
import os

QA_SYSTEM_PROMPT = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Answer in Arabic only. Make sure to avoid repeating the question."""

QA_PROMPT = """Question: {question} \nContext: {context}"""

SUMMARY_SYSTEM_PROMPT = """You are an assistant for question-answering tasks. \
Use the following retrieved context to create a summary answer to the question. \
If the context doesn't contain any relevant information, just say you don't know. \
Answer completely in Arabic."""

SUMMARY_PROMPT = """Question: {question} \nContext: {content} \nAnswer:"""

LLAMA_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

MIXTRAL_PROMPT_TEMPLATE = """"""

load_dotenv(Path("../.env"))

def get_model(model_name, temprature=0.7):
    if "gpt" in model_name:
        # TODO
        pass
    else:
        llm = Together(
            model=model_name,
            max_tokens=1024,
            temperature=temprature,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            together_api_key=os.getenv("TOGETHER_API_KEY")
        )

        llm = llm.bind(stop=["<|eot_id|>"])

    return llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(llm, retriever, format_doc_fn, prompt_template):

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain