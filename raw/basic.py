import pandas as pd

# CSV 파일 읽기
df = pd.read_csv("raw_data.csv")

# 데이터프레임의 기본 정보 출력
print(df.info())
print("-" * 100)
# # 데이터프레임의 첫 몇 줄 출력
# print(df.head())

# # 데이터프레임의 통계 요약 출력
# print(df.describe())

# "지원분야" 열의 unique value 들을 출력
print(df["지원분야"].unique())
print("-" * 100)
# 각 column의 결측치 조사
missing_values = df.isnull().sum()
print(missing_values)

print("-" * 100)
# int로 되어있는 값들의 max 값들을 출력
int_columns_max = df.select_dtypes(include="int").max()
print(int_columns_max)

# "지원분야" 별로 csv 파일로 분리해서 저장
for field in df["지원분야"].unique():
    field_df = df[df["지원분야"] == field]
    field_df.to_csv(f"{field}_data.csv", index=False)
