from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import uuid

extracted_data = load_pdf(data = 'Data/')
text_chunks =text_split(extracted_data)
embeddings = download_hugging_face_embeddings()



from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
index_name = "medicb"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Prepare documents and embeddings
ids = [str(uuid.uuid4()) for _ in text_chunks]
texts = [doc.page_content for doc in text_chunks]
vectors = embeddings(texts)

pinecone_vectors = [
    {
        "id": id_,
        "values": vector,
        "metadata": {
            "text_snippet": text[:500]
        }
    }
    for id_, vector, text in zip(ids, vectors, texts)
]


# Upsert to Pinecone
from tqdm import tqdm

batch_size = 100  # You can also try 50 if needed
for i in tqdm(range(0, len(pinecone_vectors), batch_size)):
    batch = pinecone_vectors[i:i+batch_size]
    index.upsert(vectors=batch)


print(f"✅ Successfully stored {len(pinecone_vectors)} chunks in Pinecone.")