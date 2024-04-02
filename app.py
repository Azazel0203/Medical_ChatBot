from flask import Flask, render_template, request
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
from src.prompt import *
from src.helper import pinecone_init, get_embeddings, load_vectorstore
load_dotenv()
memory = ConversationBufferWindowMemory(k=3)
app = Flask(__name__)

index = pinecone_init()
embeddings = get_embeddings()
vectorstore, PROMPT = load_vectorstore(prompt_template=prompt_template, embeddings=embeddings)
chain_type_kwargs={"prompt": PROMPT, "memory":ConversationBufferWindowMemory(memory_key="history", input_key="question")}

llm = CTransformers(model=".\model\llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.9})

question_answer_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
    chain_type_kwargs=chain_type_kwargs,
    memory=memory,
    verbose=True
)


# decorator
@app.route("/") # at this location of the webpage
#flask will open run this function
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    print("MEMORY->")
    print (memory)
    try:
        msg = request.form["msg"]
    except KeyError:
        return "Error: 'msg' field not found in the request form."
    input=msg
    print(input)
    result = question_answer_chain.invoke({"query":input})
    print ("Response: ", result['result'])
    return str(result['result'])

if __name__ =='__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
