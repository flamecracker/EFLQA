import pandas as pd
import os
import ast, textwrap

def read_csvs_from_folder(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = {}
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path,encoding='utf-8',sep='|')
        dataframes[file] = df

    return dataframes

def read_csvs_from_folder_left(book_folder_path,processed_folder_path):
    book_files = os.listdir(book_folder_path)
    processed_files = os.listdir(processed_folder_path)

    # 파일 이름에서 숫자 부분만 추출하여 집합(set)으로 저장합니다.
    book_numbers = set([file.split('.')[0] for file in book_files])
    processed_numbers = set([file.split('.')[0] for file in processed_files])

    # 'book' 폴더에는 있지만 'processed' 폴더에 없는 파일의 숫자를 찾습니다.
    unprocessed_numbers = book_numbers - processed_numbers
    csv_files = [f"{num}.csv" for num in unprocessed_numbers]
    dataframes = {}
    for file in csv_files:
        file_path = os.path.join(book_folder_path, file)
        df = pd.read_csv(file_path,encoding='utf-8',sep='|')
        dataframes[file] = df

    return dataframes

def split_question_answer(text):
    questions = []
    answers = []
    try:
        lines = text.split("\n")
    except:
        return pd.Series([questions, answers])
    

    question_parts = []

    for line in lines:
        if "Answer:" in line:
            questions.append("\n".join(question_parts).strip())
            question_parts = []
            answers.append(line.split("Answer:")[1].strip())
        else:
            question_parts.append(line.strip())
    return pd.Series([questions, answers])

def parse_string_to_list(string):
    try:
        return ast.literal_eval(string)
    except ValueError:
        return []

def wrap_text(text, width=60):
    return '\n'.join(textwrap.wrap(text, width=width))