import os
from flask import Flask, request, render_template
from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as LC_Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from pinecone import Pinecone

from src.prompt import system_prompt  # your custom prompt

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("PINECONE_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
index_name = "medicb"

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Get Pinecone index instance
index = pc.Index(index_name)

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create LangChain Pinecone vectorstore with required text_key
vectorstore = LC_Pinecone(index=index, embedding=embeddings, text_key="text")

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Initialize OpenAI chat model
llm = ChatOpenAI(
    temperature=0.4,
    max_tokens=500,
    openai_api_key=openai_key
)

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# Create conversational retrieval QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt}
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form.get("msg")
    print(f"User input: {user_input}")

    chat_history = []

    result = qa_chain.invoke({"question": user_input, "chat_history": chat_history})
    response = result.get("answer", "Sorry, I don't know the answer.")

    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)