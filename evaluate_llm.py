import pandas as pd
import os
import random
from utils.data_utils import wrap_text, parse_string_to_list

from tqdm import tqdm
from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
llama_pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

# data/processed 폴더 내의 모든 csv 파일 목록을 가져옵니다.
file_list = os.listdir('data/processed')
csv_files = [file for file in file_list if file.endswith('.csv')]

# 랜덤으로 100개의 파일을 선택합니다. (파일 수가 100개 미만일 경우 모든 파일을 선택)
#selected_files = random.sample(csv_files, min(100, len(csv_files)))

# 결과를 저장할 디렉터리 확인 및 생성
output_dir = 'data/test'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file in tqdm(csv_files, desc="Processing files"):
    df = pd.read_csv(f'data/processed/{file}', sep='|')
    df['Question'] = df['Question'].apply(parse_string_to_list)
    df['Answer'] = df['Answer'].apply(parse_string_to_list)
    results = []
    # 각 파일에 대한 결과를 저장할 빈 문자열을 초기화합니다.
    
    for index, row in df.iterrows():
        passage = row['generated_passage']
        for q, a in zip(row['Question'], row['Answer']):
            prompt = f'Given a passage and a question-answer pair, you need to determine whether the question can be answered based on the passage and if the answer is correct. \
            If both conditions are met, respond with Yes; otherwise, respond with No. The given passage and the question-answer pair are as follows: {passage}\n{q}\n{a}.\n  \
            Please ensure the output is strictly Yes or No, and kindly reformulate this prompt into polished English. Result: '
            # LLaMA 2 모델을 사용하여 질문에 대한 응답 생성
            sequences = llama_pipeline(
                prompt,
                do_sample=True,
                top_k=5,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length = 2000,
            )
            response = sequences[0]['generated_text']
            try:
                result_word = response.split('Result: ')[1].split()[0]
            except:
                result_word = None
                
            #print(result_word)
            results.append({'Passage': passage, 'Question': q, 'Answer': a, 'Model_Response': result_word})
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/{file}', index=False, sep='|')
    # 각 파일의 처리 결과를 출력합니다.