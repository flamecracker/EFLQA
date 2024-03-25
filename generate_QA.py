#import section
import pandas as pd
import os, time
from utils.data_utils import read_csvs_from_folder, split_question_answer, read_csvs_from_folder_left
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key='##########') # fill this api key number


# code for get passage data

def generate_passage_gpt(passage,word):
    model='gpt-3.5-turbo-0125'
    base=f'You are an assistant that rewrite the passage using a given vocabulary of 3000 words for English educational passages for middle school students. Here is the list of the vocabulary: {word}'
    input_message = f'Write an English educational passage for middle school students, refined to about 150 words, from the following given paragraph: {passage}'
    try:
        response = client.chat.completions.create(
            model=model, 
            messages=[
                {"role": "system", "content": base},
                {"role": "user", "content": input_message}
            ],
            temperature=0.5, 
            max_tokens=300) # 늘려야할듯
        #print(response.choices[0].message.content)
        time.sleep(5)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

def generate_QA_gpt(passage):
    model='gpt-3.5-turbo-0125'
    base=f'You are an assistant that generates pairs of questions and answers corresponding to the following types for a given text passage. Make sure you provide the answer for each question.\
        - Understand the details of the text.\
        - Identify the topic or main idea of the text.\
        - Infer the mood, the speaker, and the feelings and intentions of the characters.\
        - Understand the logical relationship of events or incidents in the text.\
        - Infer the implied meaning of words, phrases, and sentences in the text.\
        - Understand the development method or structure of the text.'
    input_message = f'Create six pairs of multiple-choice questions with five options each for the given text passage: {passage}'
    try:
        response = client.chat.completions.create(
            model=model, 
            messages=[
                {"role": "system", "content": base},
                {"role": "user", "content": input_message}
            ],
            temperature=0.5, 
            max_tokens=500) #이것도 늘려야함
        #print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":

    folder_path = 'data/passage'
    word_path = 'data/word.txt'
    processed_path = 'data/processed'
    #processed_path = 'data/processed_gpt4'
    with open(word_path, 'r') as file:
        content = file.read()
    words = content
    test_passage_dict = read_csvs_from_folder_left(folder_path,processed_path)
    for book in tqdm(test_passage_dict.keys(), desc="book processing"):
        tqdm.pandas(desc=f'{book} - Create generated_passage')
        test_passage_dict[book]['generated_passage'] = test_passage_dict[book].progress_apply(lambda row: generate_passage_gpt(row['passage'], words), axis=1)
        tqdm.pandas(desc=f'{book} - Create generated_QA')
        test_passage_dict[book]['generated_QA'] = test_passage_dict[book].progress_apply(lambda row: generate_QA_gpt(row['generated_passage']), axis=1)
        test_passage_dict[book][['Question','Answer']] = test_passage_dict[book]['generated_QA'].apply(split_question_answer)
        test_passage_dict[book].drop(columns=['passage','generated_QA'],inplace=True)
        output_path = f'{processed_path}/{book}'
        test_passage_dict[book].to_csv(output_path, sep='|', index=False)

    
