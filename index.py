from src.helper import load_pdf, text_split, download_hugging_face_embeddings
import pinecone
from langchain_community.vectorstores import Pinecone  
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()

# Get and set Pinecone API key
pine_api = os.environ.get("PINECONE_API_KEY")

# Load and process documents
extracted_data = load_pdf(data='Data/')        # Ensure this returns a string or list of strings
text_chunks = text_split(extracted_data)            # Should return list[Document] or list[str]

# Load embeddings (must match the dimension used in Pinecone)
embeddings = download_hugging_face_embeddings()     # Ensure this returns a LangChain-compatible Embeddings object

# Initialize Pinecone client
pinecone.init(api_key=pine_api, environment="us-east1-gcp")
# Create index if it doesn't exist
index_name = "medicb"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=384,
        metric="cosine"
    )


# Store documents in Pinecone via LangChain wrapper
docsearch = Pinecone.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index=index_name
)