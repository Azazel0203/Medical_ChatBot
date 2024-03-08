from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain_pinecone import PineconeVectorStore
import pinecone
from dotenv import load_dotenv
from src.prompt import *
import os
load_dotenv()

app = Flask(__name__)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
embeddings = download_hugging_face_embeddings()
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(PINECONE_INDEX_NAME)
print (index.describe_index_stats())

vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
PROMPT = PromptTemplate(template=prompt_template, input_variables=["question", "context"])
chain_type_kwargs={"prompt": PROMPT}
llm = CTransformers(model="E:\ML\generative_ai\Medical_ChatBot\model\llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.9})

question_answer_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)


# decorator
@app.route("/") # at this location of the webpage
#flask will open run this function
def index():
    return render_template('chat.html')


if __name__ =='__main__':
    app.run(debug=True)
