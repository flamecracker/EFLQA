import pandas as pd
import random

df = pd.read_csv("merged_topics_books.csv",index_col = 0)
print(df)
topic_books = {}
for index, row in df.iterrows():
    # 비어있는 Title과 Author는 건너뛴다
    if row['Title'] == "" or row['Author'] == "":
        continue
    
    # 모든 토픽 필드에 대해 처리
    for topic in row['agg_top_topic1'], row['agg_top_topic2'], row['agg_top_topic3']:
        if topic not in topic_books:
            topic_books[topic] = []
        topic_books[topic].append((row['Title'], row['Author']))

# 각 토픽별로 Title과 Author를 무작위로 3개씩 선택 (또는 가능한 만큼)
result = []
for topic in range(30):  # 토픽 0부터 29까지
    titles_authors = topic_books.get(topic, [])
    if len(titles_authors) > 3:
        titles_authors = random.sample(titles_authors, 3)
    result.append({
        "topic": topic,
        "title": [ta[0] for ta in titles_authors],
        "author": [ta[1] for ta in titles_authors]
    })

# 결과를 DataFrame으로 변환하고 CSV 파일로 저장
result_df = pd.DataFrame(result)
result_df.to_csv("topics_per_title_matched.csv", index=False)

result_df.head()

# 파일을 생성하고 데이터를 작성한다.
with open("topics_per_title.txt", "w") as file:
    for item in result:  # result는 위 코드에서 생성된 리스트를 사용한다.
        file.write(f"[{item['topic']}]\n")  # 토픽 번호를 쓴다.
        for title, author in zip(item['title'], item['author']):
            file.write(f"{title}, {author}\n")  # 제목과 저자를 쓴다.
        file.write("\n")  # 각 토픽 사이에 빈 줄을 삽입한다.