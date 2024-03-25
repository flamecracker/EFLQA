
import os
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
nltk.download('punkt')

def sample_passage(id, bookfile, passage_path, min_length=50, min_sentence=6, max_sentence=13):
    f = open(path+bookfile, 'r', encoding='utf-8')
    book = f.read()
    book = book.replace("|", " ")
    # book = book.replace(b'\xe2\x80\x9c', b'"')
    # Split paragraphs
    parag = book.split("\r\n\r\n")
    parag = book.split("\n\n\n\n")
    parag = [x.replace("\r\n", " ") for x in parag]
    parag = [x.replace("\n\n", " ") for x in parag]
    parag = [x.replace("\r", " ") for x in parag]
    parag = [x.replace("\n", " ") for x in parag]
    parag = [p for p in parag if p.find('Project Gutenberg') == -1]
    parag = [p for p in parag if p.lower().find('illustration') == -1]
    parag = [p for p in parag if p.upper() != p]
    
    # Remove paragraphs below a certain length
    parag = [p for p in parag if len(p) > min_length]

    temp = ''
    new_par = []
    length_of_sentence = 0
    start_index = 0
    for i in range(len(parag) - 1):
        length_of_sentence += len(sent_tokenize(parag[i]))
        if length_of_sentence < min_sentence:
            i += 1
            continue
        else:
            for j in range(start_index, i + 1):
                temp += parag[j]
            new_par.append(temp)
            start_index = i + 1
            temp = ''
            length_of_sentence = 0

    new_par = [p for p in new_par if len(sent_tokenize(p)) < max_sentence]

    if not new_par:
        #print(f"{bookfile} is empty.")
        return

    df = pd.DataFrame(new_par, columns=['passage'])
    df = df.applymap(lambda x: ' '.join(x.split()))
    #print(df)
    #print("##### %d #####" % id)
    df.to_csv(passage_path + str(bookfile.split('.')[0]) + '.csv', sep ='|', encoding='utf-8',index=None)


path = 'data/book/'
passage_path = 'data/passage/'
# df = extract_book_by_subject('child')
bookfiles = os.listdir(path)
#bookfiles = bookfiles[:3]

if not os.path.exists(passage_path):
    os.makedirs(passage_path)  # 폴더 생성
    print(f"Created '{passage_path}'")
else:
    print(f"'{passage_path}' already exists.")

for i, book in tqdm(enumerate(bookfiles), total=len(bookfiles)):
    sample_passage(i, book, passage_path, min_length=200, min_sentence=30, max_sentence=40) #default 200, 20, 30

