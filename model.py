from langchain_google_genai import ChatGoogleGenerativeAI
from googletrans import Translator
AI_TOKEN = "AIzaSyATVrH7HwoH71oO_Ln7zNI80pDL-Qv7QXQ"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key = AI_TOKEN
)

from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

persist_directory = '/Users/vardhansans/Downloads/ChatBot-main/database'
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key = AI_TOKEN)

vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_directory
)

retriever = vectordb.as_retriever()

from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain , RetrievalQA

# Build prompt
template = """You are a Museum ChatBot , Use the following pieces of context to answer the question at the end. If the user is having conversation with you please reply to it and For a question If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. The payment process is also handled by the bot , So give appropriate costs and booking options and if the user wants to buy/book ticket return this response '!👇🏿 Intiate Payment Gateway'.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

conversational_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever=retriever,
    memory = memory,
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
)

def start():
    translator = Translator()
    lang = input("Enter your Language preference : ")

    if lang == 'english':
        while True:
            question = input()
            if question == "bye":
                break
            result = conversational_chain({"question" : question})
            print(f"Bot : {result['answer']}\n")
    elif lang == 'hindi':
        while True:
            question = input()
            if question == "bye":
                break
            question = translator.translate(question , src="hi" , dest="en")
            result = conversational_chain({"question" : question.text})
            result = translator.translate(result['answer'] , src="en" , dest="hi")
            print(f"Bot : {result.text}\n")