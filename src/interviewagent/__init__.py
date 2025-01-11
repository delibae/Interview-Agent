import json
import os
import re
from typing import List

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from interviewagent.emb_gen import get_embeddings
from interviewagent.milvus import (
    MILVUS,
    CustomMilvusResult,
    create_collection,
    query_data,
)


def insert_data(
    data: pd.DataFrame,
    collection_name: str,
    answer_fields: List[str],
):
    for index, row in tqdm(data.iterrows(), total=len(data)):
        text_chunks = [row[field] for field in answer_fields if row[field] is not None]
        embeddings = get_embeddings(text_chunks)
        for i, embedding in enumerate(embeddings, start=1):
            MILVUS.insert(
                collection_name=collection_name,
                data=[
                    {
                        "application_id": f"{row['application_id']}",
                        "embedding": embedding,
                        "question_id": f"question_{i}",
                    }
                ],
            )


def end_to_end_insert_pipeline(
    data_file_path: str,
    collection_name: str,
    answer_fields: List[str],
):
    data = pd.read_csv(data_file_path)
    create_collection(collection_name, data_size=len(data), dim=1024)
    insert_data(data, collection_name, answer_fields)


def query_text_data(collection_name: str, query_text: str) -> List[CustomMilvusResult]:
    query_embedding = get_embeddings([query_text])
    return query_data(collection_name, query_embedding)


class ExampleInputList:
    def __init__(self, example_inputs: List[dict]):
        self.example_inputs = example_inputs

    def formatting_to_str(self, answer_eval_dict: dict) -> str:
        text = ""
        for i, example_input in enumerate(self.example_inputs):
            text += f"예제 {i+1}\n"
            print(example_input)
            for key, value in answer_eval_dict.items():
                text += f"{value}: {example_input[key]}\n"
        return text

    def to_dict(self) -> List[dict]:
        return [example_input for example_input in self.example_inputs]


def query_text_data_and_get_original_data(
    collection_name: str,
    query_text: str,
    data: pd.DataFrame,
    index_column: str = "application_id",
    answer_fields: List[str] = None,
    eval_fields: List[str] = None,
) -> ExampleInputList:
    results = query_text_data(collection_name, query_text)
    data_list = [
        data[data[index_column] == result.entity.application_id].iloc[0]
        for result in results
    ]
    data_list.sort(key=lambda x: x["average_score"])

    top_entry = data_list[0]
    middle_entry = data_list[len(data_list) // 2]
    bottom_entry = data_list[-1]

    data_list = [top_entry, middle_entry, bottom_entry]

    example_inputs = []
    for data_row in data_list:
        example_input = {
            field: data_row[field] for field in answer_fields + eval_fields
        }
        example_inputs.append(example_input)

    return ExampleInputList(example_inputs)


def generate_score_from_example_input(
    query_text: str,
    example_input_list: ExampleInputList,
    answer_eval_dict: dict,
) -> float:
    prompt = f"""
    실제 면접 데이터를 바탕으로 새로운 지원자에 대한 점수를 예측해주세요.
    json 형식으로 답변해주세요.
    description: 간략한 종합 평가입니다. (실제 면접 데이터에 근거하여 작성해주세요.)
    {json.dumps(answer_eval_dict, ensure_ascii=False, indent=4)}
    """
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url="https://api.deepinfra.com/v1/openai",
    )

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": f"실제 면접 데이터: {example_input_list.formatting_to_str(answer_eval_dict)}",
            },
            {"role": "user", "content": f"새로운 지원자: {query_text}"},
            {"role": "user", "content": prompt},
        ],
    )
    print(response.choices[0].message.content)
    matches = re.findall(r"\{.*\}", response.choices[0].message.content, re.DOTALL)
    response_dict = json.loads(matches[0])
    return response_dict
