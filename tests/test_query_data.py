import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from interviewagent import (
    generate_score_from_example_input,
    query_text_data,
    query_text_data_and_get_original_data,
)

query_text = "저는 기록물 관리에 관한 연구를 끊임없이 공부하고 있습니다. 새로운 논문, 도서를 읽으며 업무에 적용할 수 있는 방안에 대해 고심합니다. 또 같은 기록물관리전..."
# result = query_text_data(
#     "career_4_records_management_naju",
#     "저는 기록물 관리에 관한 연구를 끊임없이 공부하고 있습니다. 새로운 논문, 도서를 읽으며 업무에 적용할 수 있는 방안에 대해 고심합니다. 또 같은 기록물관리전...",
# )

# for i in result:
#     print(i)

data = pd.read_csv("datas/sample_data.csv")

eval_fields = [
    "eval1_1",
    "eval_1_basic_1_8",
    "eval_1_basic_2_8",
    "eval_1_basic_2_82",
    "eval_1_basic_2_83",
    "eval_2_ability_15",
    "eval_2_basic_1_10",
    "eval_2_basic_1_11",
    "eval_2_basic_2_10",
    "eval_2_fit_15",
]

answer_fields = ["answer_1", "answer_2", "answer_3", "answer_4", "answer_5"]

answer_eval_dict = {
    "answer_1": "질문 1 답변",
    "answer_2": "질문 2 답변",
    "answer_3": "질문 3 답변",
    "answer_4": "질문 4 답변",
    "answer_5": "질문 5 답변",
    "eval1_1": "평가원 1 1",
    "eval_1_basic_1_8": "평가원 1 basic max score 8",
    "eval_1_basic_2_8": "평가원 1 basic max score 8",
    "eval_1_basic_2_82": "평가원 1 basic max score 82",
    "eval_1_basic_2_83": "평가원 1 basic max score 83",
    "eval_2_ability_15": "평가원 2 ability max score 15",
    "eval_2_basic_1_10": "평가원 2 basic max score 10",
    "eval_2_basic_1_11": "평가원 2 basic max score 11",
    "eval_2_basic_2_10": "평가원 2 basic max score 10",
    "eval_2_fit_15": "평가원 2 fit max score 15",
}


result = query_text_data_and_get_original_data(
    collection_name="test_collection",
    query_text=query_text,
    data=data,
    answer_fields=answer_fields,
    eval_fields=eval_fields,
)
# for i in result.example_inputs:
#     print(i)

print(result.formatting_to_str(answer_eval_dict))


# print(result.formatting_to_str())

print(generate_score_from_example_input(query_text, result, answer_eval_dict))
