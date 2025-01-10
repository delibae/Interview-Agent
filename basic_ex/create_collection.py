import os

import pandas as pd
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

# Step 2: Embed and store in Zilliz
milvus = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))
index_params = MilvusClient.prepare_index_params()

# https://milvus.io/blog/select-index-parameters-ivf-index.md
# 4 * sqrt(n)
nlist = int(((df.shape[0]) ** 0.5) * 4)
index_params.add_index(
    field_name="embedding",
    metric_type="COSINE",
    index_type="IVF_FLAT",
    index_name="vector_index",
    params={"nlist": nlist},
)

schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=False,
)
schema.add_field(
    field_name="id",
    datatype=DataType.INT64,
    is_primary=True,
    auto_id=True,
    description="id",
)
schema.add_field(
    field_name="application_id",
    datatype=DataType.VARCHAR,
    description="application id",
    max_length=100,
)
schema.add_field(
    field_name="question_id",
    datatype=DataType.VARCHAR,
    description="question id",
    max_length=100,
)
schema.add_field(
    field_name="embedding",
    datatype=DataType.FLOAT_VECTOR,
    dim=1024,
    description="embedding vector",
)

milvus.create_collection(
    collection_name="career_4_records_management_naju",
    dimension=1024,
    index_params=index_params,
    schema=schema,
)
