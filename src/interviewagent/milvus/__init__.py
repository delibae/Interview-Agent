import os
from typing import List

from openai import BaseModel
from pymilvus import DataType, MilvusClient

MILVUS = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))


def create_collection(collection_name: str, data_size: int, dim: int):
    # Step 2: Embed and store in Zilliz
    index_params = MilvusClient.prepare_index_params()

    # https://milvus.io/blog/select-index-parameters-ivf-index.md
    # 4 * sqrt(n)

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
        dim=dim,
        description="embedding vector",
    )

    nlist = int((data_size**0.5) * 4)
    index_params.add_index(
        field_name="embedding",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        index_name="vector_index",
        params={"nlist": nlist},
    )

    MILVUS.create_collection(
        collection_name=collection_name,
        dimension=dim,
        index_params=index_params,
        schema=schema,
    )


class InterviewFields(BaseModel):
    id: str
    application_id: str
    question_id: str


class CustomMilvusResult:
    def __init__(self, id: str, distance: float, entity: dict):
        self.id = id
        self.distance = distance
        self.entity = InterviewFields(**entity)

    def __str__(self):
        return f"id: {self.id}, distance: {self.distance}, entity: {self.entity}"


def query_data(
    collection_name: str,
    query_embedding: List[List[float]],
    limit: int = 5,
    output_fields: List[str] = ["id", "application_id", "question_id"],
) -> List[CustomMilvusResult]:
    assert len(query_embedding) == 1
    results = MILVUS.search(
        collection_name=collection_name,
        data=query_embedding,
        limit=limit,
        output_fields=output_fields,
    )
    return [CustomMilvusResult(**result) for result in results[0]]
