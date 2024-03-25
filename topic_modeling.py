import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re
import spacy

# Spacy의 영어 모델 로드
print(spacy.require_gpu())
spacy.prefer_gpu()
nlp = spacy.load('en_core_web_trf')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# 데이터 정제 함수
def cleansing(book):
    book = book.replace("|", " ")
    paragraphs = book.split("\n\n")  # 간단하고 일관된 구분자 사용
    paragraphs = [x.replace("\r\n", " ").replace("\n", " ") for x in paragraphs]
    paragraphs = [p for p in paragraphs if 'project gutenberg' not in p.lower()]
    paragraphs = [p for p in paragraphs if 'illustration' not in p.lower()]
    paragraphs = [p for p in paragraphs if 'chapter' not in p.lower()]
    paragraphs = [p for p in paragraphs if 'index' not in p.lower()]
    paragraphs = [p.strip() for p in paragraphs if p.upper() != p]  # 대문자만 있는 문단 제외
    return paragraphs

def assign_topics_to_documents(model, vectorizer, documents, bookfiles):
    # 각 문서에 대한 주제 분포를 계산
    topic_distributions = model.transform(vectorizer.transform(documents))
    
    # 각 문서에 대해 가장 높은 점수를 가진 주제 찾기
    dominant_topics = np.argmax(topic_distributions, axis=1)
    
    # 각 파일과 그에 해당하는 주제를 출력
    for bookfile, topic in zip(bookfiles, dominant_topics):
        print(f"File: {bookfile}, Dominant Topic: {topic + 1}")

#텍스트 정제
def preprocess_text(text):
    # 텍스트를 모두 소문자로 변환
    text = text.lower()
    doc = nlp(text)
    
    tokens = []
    for token in doc:
        # 고유명사와 spaCy 내장 불용어 거르기
        if token.lemma_ not in custom_stop_words or not token.is_stop:
            if (token.ent_type_ not in ['PERSON', 'GPE']) and (token.pos_ not in ['PUNCT','PRON','PROPN']):
                # 토큰의 기본형과 품사 태그 결합
                token_pos = f"{token.lemma_}_{token.pos_}"
                #print(token_pos,token.is_stop)
                # 특수 문자 제거
                token_pos_clean = re.sub(r'[^a-zA-Z0-9\_]', '', token_pos)
                # 길이가 3 미만인 단어 제외
                if len(token_pos_clean.split('_')[0]) > 2:
                    tokens.append(token_pos_clean)
    
    return ' '.join(tokens)


def split_into_segments(text, max_length=5000):
    sentences = sent_tokenize(text)
    current_chunk = []
    current_length = 0
    chunks = []
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    
    # 마지막 청크 추가
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
tqdm.pandas()

# 데이터 로딩 및 전처리

path = 'data/book'
bookfiles = os.listdir(path)
documents = []
titles = []

max_length = 1000

for book in tqdm(bookfiles, desc="Loading and Cleansing Books"):
    with open(os.path.join(path, book), 'r', encoding='utf-8') as f:
        book_content = f.read()
        paragraphs = cleansing(book_content)  # 정제 함수 적용
        document = ' '.join(paragraphs)
        # 문서 길이가 5,000자를 초과하는 경우 분리
        segments = split_into_segments(document, max_length)
        for segment in segments:
            documents.append(segment)
            titles.append(book)
        
print("Total Documents num: ", len(documents))

# 반복문이 끝난 후 한 번에 DataFrame 생성
df = pd.DataFrame({'title': titles, 'document': documents})

nltk.download('stopwords')
stop_words = stopwords.words('english')
custom_stop_words = stop_words + ['contents', 'introduction', 'end', 'illustrations', 'illustration',
                                  'freddie','fred','alice','tommy','anne','dorothy','heidi','dick','jane','billie','elsie',
                                  'polly', 'jasper', 'grace', 'dotty', 'dora', 'marian', 'joel', 'harriet', 'josie', 'darrin',
                                  'frank', 'betty', 'flossie', 'janice', 'bessie', 'ozma', 'helen','negro','marjorie','eleanor',
                                  'philip','ally','billy','rollo','phronsie','wiggily','robin','say']  # 커스텀 불용어 추가 ,'man','woman'
for word in custom_stop_words:
    #print(word)
    nlp.vocab[word].is_stop = True

#df = df.sample(n=100, random_state=122)

df['clean_doc'] = df['document'].progress_apply(preprocess_text)
df.to_csv('processed_text_TM_30.csv',index=False)


# 주제 출력

def get_topics(components, feature_names, n=7):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(3)) for i in topic.argsort()[:-n - 1:-1]])

#sampled_df['clean_doc'] = sampled_df['document'].progress_apply(preprocess_text)

#df = pd.read_csv('processed_text_TM_30.csv')
# 결측값을 빈 문자열로 대체
#df['clean_doc'] = df['clean_doc'].fillna('')
# TF-IDF 벡터화
print("vectorizing start.")
vectorizer = TfidfVectorizer(max_features=3000, max_df=0.5, smooth_idf=True)
X = vectorizer.fit_transform(df['clean_doc'])


#### LDA 토픽모델링

from sklearn.decomposition import LatentDirichletAllocation
lda_model = LatentDirichletAllocation(n_components=30,learning_method='online',random_state=777,max_iter=100)#10
print("LDA model report")
lda_top = lda_model.fit_transform(X)
terms = vectorizer.get_feature_names_out()
get_topics(lda_model.components_,terms)

top_topics_per_document = np.argsort(lda_top, axis=1)[:, -3:]

topics_df = pd.DataFrame({
    'title': df['title'],
    'top_topic1': top_topics_per_document[:, 2],
    'top_topic2': top_topics_per_document[:, 1],
    'top_topic3': top_topics_per_document[:, 0]
})
# 책 제목별로 가장 많이 할당된 상위 3개의 토픽을 찾기
def aggregate_top_topics(row):
    topics = row[['top_topic1', 'top_topic2', 'top_topic3']].values.flatten()
    top_3_topics = pd.Series(topics).value_counts().index[:3]
    return pd.Series(top_3_topics, index=['agg_top_topic1', 'agg_top_topic2', 'agg_top_topic3'])

agg_topics_per_title = topics_df.groupby('title').apply(aggregate_top_topics).reset_index()

agg_topics_per_title.to_csv('topics_per_title.csv', index=False, encoding='utf-8')

print(agg_topics_per_title)

import pyLDAvis
import pyLDAvis.lda_model
#pyLDAvis.enable_notebook()

lda_visualization = pyLDAvis.lda_model.prepare(lda_model, X, vectorizer, mds='tsne')

# HTML 파일로 저장
pyLDAvis.save_html(lda_visualization, 'lda_visualization.html')

print("LDA Visualization completed.")