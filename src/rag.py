import os
from typing import List, Tuple
from operator import itemgetter
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from langchain.prompts.prompt import PromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_pinecone import PineconeVectorStore

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_INDEX_NAME=os.getenv("PINECONE_INDEX_NAME")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vectorstore = PineconeVectorStore.from_existing_index(
    PINECONE_INDEX_NAME, embeddings
)
retriever = vectorstore.as_retriever()

parser = StrOutputParser()

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")

from langchain.chains import ConversationalRetrievalChain

bot = ConversationalRetrievalChain.from_llm(
    model, retriever, memory=memory, verbose=False
)

from langchain.prompts.prompt import PromptTemplate

_template = """Dada a conversa a seguir e a pergunta que a acompanha, reformule a pergunta de acompanhamento para ser uma pergunta independente, sempre em português.

Chat History:
{chat_history}
Pergunta que a acompanha: {question}
Pergunta independente:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

from langchain_core.prompts import ChatPromptTemplate

template = """Responda a pergunta baseado somente nesse contexto:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

from langchain_core.messages import  get_buffer_string
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)

standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | model
    | parser,
}

retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}

final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}

answer = {
    "answer": final_inputs | ANSWER_PROMPT | model,
    "docs": itemgetter("docs"),
}

def response(answer):
  a = answer['docs']
  return f'{answer["answer"].content}  Fonte: {a[0].metadata["source"]}, página {a[0].metadata["page"]}.'

chain = loaded_memory | standalone_question | retrieved_documents | answer | (response)


