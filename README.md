## Medical Chatbot with Langchain and Pinecone

This project builds a medical chatbot that retrieves information from a medical PDF book and utilizes Langchain for processing and Pinecone for efficient information retrieval.

###  Features

* **Medical Knowledge Base:** Extracts and organizes medical information from a PDF book.
* **Langchain Integration:** Uses Langchain libraries to process user queries and match them with relevant information from the knowledge base. 
    * Specifically, Langchain's text processing capabilities will be leveraged to clean and prepare the medical text for further analysis.
* **Llama2 Embeddings:** Employs the powerful Llama2 model from Hugging Face to generate contextual embeddings for both user queries and medical text snippets. This allows for semantic matching and retrieval of relevant information even when phrased differently.
* **Pinecone Vector Database:** Stores the generated embeddings in a Pinecone vector database for efficient retrieval. This enables fast and scalable search of the medical knowledge base.
* **Chatbot Interface:** Provides a user-friendly interface (text-based or potentially voice-based) for users to interact with the chatbot and ask medical questions.
* **HTML and CSS Design:** The chat interface is designed using HTML and styled using CSS to provide an intuitive and visually appealing user experience.
* **Flask Backend:** The application backend is built using Flask, a lightweight web framework for Python, to handle user requests and interact with the chatbot.
* **Real-time Messaging:** Users can communicate with the chatbot in real-time, receiving instant responses to their queries.

###  Technical Stack

* **Langchain:** A Python library for Natural Language Processing (NLP) workflows, offering modules for text processing, embedding generation, and information retrieval.
* **Hugging Face Transformers:** Provides access to pre-trained NLP models like Llama2 for generating contextual embeddings.
* **Pinecone:** A vector database service enabling efficient storage and retrieval of high-dimensional data like embeddings.
* **Additional Libraries:** Depending on the chosen interface (text-based or voice-based), additional libraries like NLTK or spaCy might be used for further text processing and chatbot functionalities.
* **Flask:** A lightweight web framework for Python used to develop the chatbot's interface and handle user requests.

###  Workflow

1. **Knowledge Base Creation:**

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_knowledge_base(pdf_path):
  # Load PDF text
  loader = PyPDFLoader(pdf_path, glob="*.pdf")
  text_data = loader.load()

  # Text processing and chunking
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
  text_chunks = text_splitter.split_documents(text_data)

  # Download Llama2 embeddings (or your chosen model)
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

  # Generate embeddings for each text chunk
  # ... (code to generate embeddings for each chunk using embeddings object)

  # Store text chunks and embeddings in a data structure (e.g., list of dictionaries)
  knowledge_base = []
  for i, chunk in enumerate(text_chunks):
    chunk_embedding = embeddings.encode(chunk)  # Generate embedding for the chunk
    knowledge_base.append({
      "text": chunk,
      "embedding": chunk_embedding
    })
  return knowledge_base
```

2. **Pinecone Integration:**

```python
from langchain_pinecone import PineconeVectorStore
from pinecone.data.index import Index
from dotenv import load_dotenv
import os

def store_knowledge_base_in_pinecone(knowledge_base):
  load_dotenv()
  PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
  PINECONE_ENV = os.getenv("PINECONE_ENV")
  PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

  # Connect to Pinecone
  pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
  index = pc.Index(PINECONE_INDEX_NAME)

  # Extract text and embeddings from knowledge base
  text_data = [kb["text"] for kb in knowledge_base]
  embeddings = [kb["embedding"] for kb in knowledge_base]

  # Store embeddings in Pinecone
  PineconeVectorStore.from_documents(text_data, embeddings, index_name=PINECONE_INDEX_NAME)

  print(f"Knowledge base stored in Pinecone index: {PINECONE_INDEX_NAME}")
```

3. **Chatbot Interface (placeholder):**

```python
# This section is a placeholder as the full chatbot development requires additional libraries
# like Rasa or Dialogflow. Here's a basic outline to illustrate the concept.

def chatbot_loop():
  while True:
    user_query = input("Ask me a medical question (or type 'quit' to exit): ")
    if user_query.lower() == "quit":
      break

    # Process user query (similar to text processing in knowledge base creation)
    processed_query = # (code to clean and process the user query)

    # Generate embedding for the user query
    query_embedding = embeddings.encode(processed_query)

    # Retrieve similar text snippets from Pinecone using query embedding
    similar_results = retrieve_from_pinecone(query_embedding)

    # Extract and present relevant information to the user
    if similar_results:
      for result in similar_results:
        print(f"Relevant Information: {result['text']}")
    else:
      print("Sorry, I couldn't find any information related to your question.")
```


###  Benefits

* **Improved Medical Information Access:** Users can easily access and understand medical information from the PDF book through a user-friendly interface.
* **Semantic Search:** Llama2 embeddings allow for more accurate retrieval of relevant information even when user queries are phrased differently from the actual text in the book.
* **Scalability:** Pinecone enables efficient information retrieval as the knowledge base grows.

###  Further Considerations

* **Medical Disclaimer:**  It's crucial to clearly state that this chatbot is for informational purposes only and should not be used as a substitute for professional medical advice. Users should always consult a healthcare professional for diagnosis and treatment. 
* **Data Security:**  Ensure proper security measures are implemented when handling and storing medical information.
* **Model Selection:** While Llama2 is a powerful model, consider exploring other medical domain-specific models from Hugging Face that might be better suited for the specific medical information in the PDF book.
* **Chatbot Development:**  Developing a robust chatbot interface requires additional libraries like Rasa or Dialogflow for handling conversation flow and natural language understanding.

This project showcases the potential of Langchain and Pinecone in creating a medical chatbot that offers an accessible and efficient way to access and understand medical information. Remember to adapt and expand upon this concept to fit the specific needs of your medical PDF book and desired functionalities.

## Installation

To set up the LangChain Pinecone Vector Store project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/<username>/<repository>.git
    cd <repository>
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Configure environment variables:

    Create a `.env` file in the root directory and specify the following variables:

    ```plaintext
    PINECONE_API_KEY=<your_pinecone_api_key>
    PINECONE_ENV=<pinecone_environment>
    PINECONE_INDEX_NAME=<pinecone_index_name>
    ```

## Usage

To store vectors in the Pinecone Vector Database, execute the following command:

```bash
python store_vectors.py
```

## Contributing

Contributions to the LangChain Pinecone Vector Store project are encouraged and appreciated! If you have ideas for enhancements, bug fixes, or new features, please submit a pull request. Be sure to follow the contribution guidelines outlined in the repository.

## License

This project is licensed under the [MIT License](LICENSE), which permits unrestricted use, distribution, and modification, subject to the terms and conditions specified in the license agreement.
