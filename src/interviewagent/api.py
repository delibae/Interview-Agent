import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from interviewagent import (
    generate_score_from_example_input,
    query_text_data_and_get_original_data,
)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


@app.post("/api/v1/get-interview-score")
async def get_interview_score(request: Request):
    request_data = await request.json()
    query_text = request_data["query_text"]
    fetched_examples = query_text_data_and_get_original_data(
        "career_4_records_management_naju",
        query_text,
        data,
    )
    score_result = generate_score_from_example_input(
        query_text,
        fetched_examples,
    )

    result = {
        "score": score_result,
        "examples": fetched_examples.to_dict(),
    }
    return result


if __name__ == "__main__":
    data = pd.read_csv("datas/career_4_records_management_naju.csv")
    uvicorn.run(app, host="0.0.0.0", port=1819)
