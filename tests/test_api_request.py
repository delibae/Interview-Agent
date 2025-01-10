import requests

url = "http://localhost:1819/api/v1/get-interview-score"

response = requests.post(
    url,
    json={
        "query_text": "저는 기록물 관리에 관한 연구를 끊임없이 공부하고 있습니다. 새로운 논문, 도서를 읽으며 업무에 적용할 수 있는 방안에 대해 고심합니다. 또 같은 기록물관리전..."
    },
)

print(response.json())
