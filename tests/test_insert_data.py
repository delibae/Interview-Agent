import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from interviewagent import end_to_end_insert_pipeline

# Define the file path and parameters
data_file_path = "datas/sample_data.csv"
collection_name = "test_collection"
answer_fields = ["answer_1", "answer_2", "answer_3", "answer_4", "answer_5"]
# eval_fields = [
#     "eval1_1",
#     "eval_1_basic_1_8",
#     "eval_1_basic_2_8",
#     "eval_1_basic_2_82",
#     "eval_1_basic_2_83",
#     "eval_2_ability_15",
#     "eval_2_basic_1_10",
#     "eval_2_basic_1_11",
#     "eval_2_basic_2_10",
#     "eval_2_fit_15",
# ]

# Call the function
end_to_end_insert_pipeline(data_file_path, collection_name, answer_fields)

# Print a simple confirmation message
print("Data insertion pipeline executed successfully.")
