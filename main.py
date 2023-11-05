from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sortedcontainers import SortedDict
from nltk.stem import PorterStemmer
import json
import nltk
import string
import re
import ir_datasets
import time
import spacy
from multiprocessing import Pool, cpu_count

class Preprocess():
    def __init__(self, docs : list[str]) -> None:
        self.docs = docs
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.translator = str.maketrans('','',string.punctuation)
        self.nlp = spacy.load('en_core_web_sm')

    def preprocess(self, text : str) -> str:
        words = word_tokenize(text)
        filtered_words = [
            self.ps.stem(word.lower()) 
            for word in words if word.lower() not in self.stop_words and word.lower().isalpha()
        ]
        return ' '.join(filtered_words)
    
    def process_chunk(self, chunk):
        return self.preprocess(chunk)
    
    def run_fast_parallel(self):

        num_processes = cpu_count()  
        chunk_size = len(self.docs) // num_processes

        #chunks = [self.docs[i:i+chunk_size] for i in range(0, len(self.docs), chunk_size)]

        with Pool(processes=num_processes) as pool:
            processed_chunks = pool.map(self.process_chunk, self.docs)

        self.docs = processed_chunks
        #self.docs = [doc for chunk in processed_chunks for doc in chunk]
    
    def get_docs(self):
        return self.docs
    
    def run_fast(self):
        for idx, doc in enumerate(self.docs):
            preprocessed_text = self.preprocess(doc)
            self.docs[idx] = preprocessed_text
    
    def save_to_file(self):
        with open('output.json','w') as file:
            json.dump(self.docs,file)

class Indexer():
    def __init__(self, docs : list[str]) -> None:
        self.docs = docs
        self.dict = {}

    def create_index(self):
        for docId, doc in enumerate(self.docs):
            words = word_tokenize(doc)
            temp_dict = {}

            for word in words:
                if temp_dict.get(word):
                    temp_dict[word] += 1
                else:
                    temp_dict[word] = 1
            
            for keys in temp_dict.keys():
                if self.dict.get(keys):
                    self.dict[keys].append([docId,temp_dict[keys]])
                else:
                    self.dict[keys] = [[docId,temp_dict[keys]]]

    def view_sample_dict(self):
        for i, keys in enumerate(self.dict.keys()):
            if i >= 10:
                break
            print(keys, end= " : ")
            for i in range(0,10):
                print(self.dict[keys][i], end=",")
    
    def save_to_file(self):
        sorted_keys = sorted(self.dict.keys())
        with open('index.txt','w') as file:
            for key in sorted_keys:
                file.write(key + ":")
                posting_list_length = len(self.dict[key])
                for idx, posting_entry in enumerate(self.dict[key]):
                    docId = str(posting_entry[0])
                    term_freq = str(posting_entry[1])
                    file.write(docId + '/' + term_freq)
                    if idx != posting_list_length - 1 :
                        file.write(' ')
                file.write('\n')
        print("Index written to index.txt")


dataset = ir_datasets.load("beir/trec-covid")

texts = []
N = len(dataset.docs)
print(N)

texts = [ docs[1] for docs in dataset.docs_iter() if len(docs[1]) > 0 ]


pre_processor = Preprocess(texts)

start_time = time.time()
#pre_processor.run_fast_parallel()
end_time = time.time()


execution_time = end_time - start_time

print('\n')

if execution_time >= 60 :
    execution_time /= 60
    print(f'Execution time : {execution_time:.6f} minutes ')
else:
    print(f'Execution time : {execution_time:.6f} seconds')

processed_docs = pre_processor.get_docs()

#pre_processor.save_to_file()


with open('output.json','r') as file:
    texts = json.load(file)
    indexor = Indexer(texts)
    indexor.create_index()
    indexor.save_to_file()