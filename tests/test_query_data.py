import pandas as pd
from dotenv import load_dotenv

from interviewagent import (
    generate_score_from_example_input,
    query_text_data,
    query_text_data_and_get_original_data,
)

load_dotenv()

query_text = "저는 기록물 관리에 관한 연구를 끊임없이 공부하고 있습니다. 새로운 논문, 도서를 읽으며 업무에 적용할 수 있는 방안에 대해 고심합니다. 또 같은 기록물관리전..."
# result = query_text_data(
#     "career_4_records_management_naju",
#     "저는 기록물 관리에 관한 연구를 끊임없이 공부하고 있습니다. 새로운 논문, 도서를 읽으며 업무에 적용할 수 있는 방안에 대해 고심합니다. 또 같은 기록물관리전...",
# )

# for i in result:
#     print(i)

data = pd.read_csv("datas/career_4_records_management_naju.csv")

result = query_text_data_and_get_original_data(
    "career_4_records_management_naju",
    "저는 기록물 관리에 관한 연구를 끊임없이 공부하고 있습니다. 새로운 논문, 도서를 읽으며 업무에 적용할 수 있는 방안에 대해 고심합니다. 또 같은 기록물관리전...",
    data,
)

print(result)

print(result.formatting_to_str())

print(generate_score_from_example_input(query_text, result))
