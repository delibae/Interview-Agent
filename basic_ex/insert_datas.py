import os

import pandas as pd
import requests
from dotenv import load_dotenv
from pymilvus import DataType, FieldSchema, MilvusClient
from tqdm import tqdm

load_dotenv()
# Step 1: Load CSV and prepare data
df = pd.read_csv("datas/경력(4급) 기록물관리(나주)_data.csv")

# Extract relevant columns
columns_of_interest = [
    "수험번호",
    "질문1_답변",
    "질문2_답변",
    "질문3_답변",
    "질문4_답변",
]
data = df[columns_of_interest]

milvus = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))


# Function to get embeddings from Jina API
def get_embeddings(text_chunks):
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('JINA_API_KEY')}",
    }
    payload = {
        "model": "jina-embeddings-v3",
        "task": "text-matching",
        "late_chunking": False,
        "dimensions": 1024,
        "embedding_type": "float",
        "input": text_chunks,
    }
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()
    return [item["embedding"] for item in response_data["data"]]


for index, row in tqdm(data.iterrows()):
    text_chunks = [
        row["질문1_답변"],
        row["질문2_답변"],
        row["질문3_답변"],
        row["질문4_답변"],
    ]
    embeddings = get_embeddings(text_chunks)

    for i, embedding in enumerate(embeddings, start=1):
        milvus.insert(
            collection_name="career_4_records_management_naju",
            data=[
                {
                    "application_id": f"{row['수험번호']}",
                    "embedding": embedding,
                    "question_id": f"question_{i}",
                }
            ],
        )
