import heapq
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json
import string
import ir_datasets
import math
import spacy
import numpy as np
from multiprocessing import Pool, cpu_count
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Preprocess():
    def __init__(self, docs: list[str]) -> None:
        self.docs = docs
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.translator = str.maketrans('', '', string.punctuation)
        #self.nlp = spacy.load('en_core_web_sm')

    def preprocess(self, text: str) -> str:
        words = word_tokenize(text)
        filtered_words = [
            self.ps.stem(word.lower())
            for word in words if word.lower() not in self.stop_words and word.lower().isalpha()
        ]
        return ' '.join(filtered_words)

    def process_chunk(self, chunk):
        return self.preprocess(chunk)

    def run_fast_parallel(self):
        print("Processing the documents.")
        num_processes = cpu_count()
        chunk_size = len(self.docs) // num_processes

        # chunks = [self.docs[i:i+chunk_size] for i in range(0, len(self.docs), chunk_size)]

        with Pool(processes=num_processes) as pool:
            processed_chunks = pool.map(self.process_chunk, self.docs)

        self.docs = processed_chunks

        print(len(self.docs))
        print("Total number of docs after processing : ", len(self.docs))

    def get_docs(self):
        return self.docs

    def run_fast(self):
        for idx, doc in enumerate(self.docs):
            preprocessed_text = self.preprocess(doc)
            self.docs[idx] = preprocessed_text

    def save_to_file(self):
        print("Saving to output.json")
        with open('output.json', 'w') as file:
            json.dump(self.docs, file)


class Indexer():
    def __init__(self, fileName: str) -> None:
        self.documents = None
        self.inverted_index = {}
        self.document_norm = None
        with open(fileName, 'r') as file:
            self.documents = json.load(file)
            self.document_norm = np.zeros(len(self.documents))

    def createIndex(self):
        document_freq = {}

        print("Creating Index")

        for doc_id, text in enumerate(self.documents):
            words = text.split()
            term_frequency = Counter(words)

            for word, count in term_frequency.items():
                if self.inverted_index.get(word):
                    self.inverted_index[word].append([doc_id, count])
                else:
                    self.inverted_index[word] = [[doc_id, count]]

                document_freq[word] = document_freq.get(word, 0) + 1

        print("Index successfully created, calculating the norm of documents.")

        for word, posting_list in self.inverted_index.items():
            idf = math.log10(len(self.documents)/document_freq[word])
            for (doc_id, tf) in posting_list:
                weight = (1 + math.log10(tf))*(idf)
                self.document_norm[doc_id] += weight**2

        self.document_norm = self.document_norm**(0.5)

        print("Norm of the documents is : \n", self.document_norm)

    def view_sample_dict(self):
        for i, keys in enumerate(self.dict.keys()):
            if i >= 10:
                break
            print(keys, end=" : ")
            for i in range(0, 10):
                print(self.dict[keys][i], end=",")

    def save_to_file(self):
        sorted_keys = sorted(self.inverted_index.keys())
        with open('index.txt', 'w') as file:
            for key in sorted_keys:
                file.write(key + " ")
                posting_list_length = len(self.inverted_index[key])
                for idx, posting_entry in enumerate(self.inverted_index[key]):
                    docId = str(posting_entry[0])
                    term_freq = str(posting_entry[1])
                    file.write(docId + '/' + term_freq)
                    if idx != posting_list_length - 1:
                        file.write(' ')
                file.write('\n')

        print("Index written to index.txt")
        print("Saving norms of document vectors to a file.")

        with open('document_norm.pkl', 'wb') as file:
            pickle.dump(self.document_norm, file)


class SearchEngine():
    def __init__(self, indexFile) -> None:
        self.indexFile = indexFile
        self.postingList = {}
        self.total_docs = set()
        self.N = 0
        self.length = None  # An array that stores the length/number of words in each doc
        self.scores = None
        self.preprocessor = None
        self.vectorizer = None
        self.tfIdfMatrix = None
        self.document_norm = None
        self.vocabulary = []
        self.query = None
        self.doc_id_mapping = {}
        dt = ir_datasets.load("beir/trec-covid")
        self.docs = dt.docs
        self.loadIndex()

    def loadIndex(self):
        if len(self.postingList) != 0:
            return
        else:
            print("Loading the index into memory")

            with open(self.indexFile, 'r') as file:
                while (True):
                    line = file.readline()

                    if len(line) == 0:
                        break

                    tokens = line.split()

                    posting_list_len = len(tokens) - 1
                    posting_list = [None]*posting_list_len

                    for i in range(1, len(tokens)):
                        posting_entry = tokens[i]
                        posting_entry = posting_entry.split('/')
                        docId = int(posting_entry[0])
                        term_frequency = int(posting_entry[1])
                        # type: ignore
                        posting_list[i-1] = (docId, term_frequency)

                    word = tokens[0]
                    self.postingList[word] = posting_list

            print('Loading document norm')
            with open('document_norm.pkl', 'rb') as file:
                self.document_norm = pickle.load(file)
                self.N = len(self.document_norm)
                self.document_norm = np.array(self.document_norm)

            # zero_indices = np.where(self.document_norm == 0)
            # print(f'Encountered zero norms : {zero_indices}')
            self.scores = np.zeros(self.N, dtype=np.float16)

            dt = ir_datasets.load("beir/trec-covid")

            nonempty_idx = [idx for idx, (_, text, _, _, _) in
                            enumerate(dt.docs_iter()) if len(text) > 0]
            self.doc_id_mapping = {id1: id2 for id1,
                                   id2 in enumerate(nonempty_idx)}

            print("Index loaded to memory")

    def searchQuery(self, query: str, k=10):

        self.query = query
        self.loadIndex()
        self.preprocessor = Preprocess([query])
        self.preprocessor.run_fast()

        (processed_query,) = self.preprocessor.get_docs()

        return self.top_k_docs(processed_query, k)

    def top_k_cosine(self, query, k):
        """ 
            Uses TfidfVectorizer and cosine similarity from scikit-learn.
            Currently using it only for experiment
        """
        cosine_similarities = None

        with open('output.json', 'r') as file:
            corpus = json.load(file)

            if self.vectorizer == None:
                print("Creating vectorizer")
                self.vectorizer = TfidfVectorizer()
                self.tfIdfMatrix = self.vectorizer.fit_transform(corpus)
                print('Successfully created vectorizer and fitted the vocabulary')

            tfIdfQueryMatrix = self.vectorizer.transform([query])
            cosine_similarities = cosine_similarity(
                tfIdfQueryMatrix, self.tfIdfMatrix)[0]

        similarities_list = list(enumerate(cosine_similarities))

        sorted_similarities_list = sorted(
            similarities_list, key=lambda x: x[1], reverse=True)

        sorted_K = sorted_similarities_list[:k]

        return sorted_K

    def get_results(self, ranked_docs):
        results = []
        for (doc_id, score) in ranked_docs:
            doc_id = int(doc_id)
            title = self.docs[self.doc_id_mapping[doc_id]][2]
            body = self.docs[self.doc_id_mapping[doc_id]][1]
            results.append({"title" : title, "score" : float(score), "body" : body})
        return results
            
    def display_result(self, ranked_docs):

        dataset = ir_datasets.load("beir/trec-covid")
        docs = dataset.docs

        for (doc_id, score) in ranked_docs:
            doc_id = int(doc_id)
            title = docs[self.doc_id_mapping[doc_id]][2]
            text = docs[self.doc_id_mapping[doc_id]][1]
            print('------------------------------------------------------')
            print()
            print(title, '[', score, ']', '[', doc_id, ']')
            print(text)
            print()
            print('------------------------------------------------------')

        res = input("Do you wish to give feedback? [y/n]: ")

        if res == 'y':
            relevant_docs = input("Enter relevant documents: ")
            relevant_docs = [ranked_docs['doc_id']
                             [int(item)] for item in relevant_docs.split()]

            non_relevant_docs = input("Enter non relevant documents: ")
            non_relevant_docs = [ranked_docs['doc_id'][int(item)]
                                 for item in non_relevant_docs.split()]

            self.preprocessor = Preprocess([self.query])
            self.preprocessor.run_fast()
            query = self.preprocessor.get_docs()[0]

            new_query_vector = self._relevance_feedback_builtin(
                query, relevant_docs, non_relevant_docs)

            print(new_query_vector)

    def top_k_docs(self, query, k):

        temp = list(Counter(query.split()).items())
        query_terms = [term for term, _ in temp]
        query_weights = np.array(
            [weight for _, weight in temp], dtype=np.float16)

        query_weights /= np.linalg.norm(query_weights)

        for (term, weight) in zip(query_terms, query_weights):
            posting_list = self.postingList[term]

            idf = math.log10(self.N/len(posting_list))

            for (doc_id, _tf) in posting_list:
                tf = 1 + math.log10(_tf)
                weight = tf*idf
                self.scores[doc_id] += weight*weight

        self.scores = self.scores/self.document_norm

        docid_score = np.fromiter(enumerate(self.scores),
                                  dtype=[('doc_id', np.int32), ('score', np.float16)])

        top_k_indices = heapq.nlargest(
            k, range(self.N), key=lambda i: docid_score['score'][i])

        top_k_documents = docid_score[top_k_indices]

        self.scores.fill(0)

        return top_k_documents

    def _relevance_feedback_builtin(self, query, relevant_docid, non_relevant_docid):
        beta = 0.75
        gamma = 0.15

        n1 = len(relevant_docid)
        n2 = len(non_relevant_docid)
        N = n1 + n2

        documents = [None]*N

        with open('output.json', 'r') as file:
            corpus = json.load(file)
            idx = 0
            for id in relevant_docid:
                documents[idx] = corpus[id]
                idx += 1
            for id in non_relevant_docid:
                documents[idx] = corpus[id]
                idx += 1

        vectorizer = TfidfVectorizer()

        documents.append(query)

        tfidf = vectorizer.fit_transform(documents)

        query_vector = tfidf[N:]
        tfidf = tfidf[:N]

        document_rmatrix = tfidf[:n1]
        document_nrmatrix = tfidf[n1:]

        centroid_diff = beta * \
            np.mean(document_rmatrix, axis=0) - gamma * \
            np.mean(document_nrmatrix, axis=0)

        centroid_diff.clip(min=0, out=centroid_diff)

        new_query_vector = query_vector + centroid_diff


        new_query_vector = np.array(new_query_vector).flatten()

        print(new_query_vector.shape)

        terms = vectorizer.get_feature_names_out()

        term_weight_vector = np.zeros(
            len(terms), dtype=[('terms', '<U50'), ('weights', np.float16)])

        for i, (term, weight) in enumerate(zip(terms, new_query_vector)):
            term_weight_vector['terms'][i] = term
            # print(weight)
            term_weight_vector['weights'][i] = weight

        sorted_indices = heapq.nlargest(
            len(query) + 2, range(len(terms)), key=lambda i: term_weight_vector['weights'][i])

        return term_weight_vector[sorted_indices]

    def relevanceFeedBack(self, query, queryVector, relevantDocIds, nonRelevantDocIds, terms):
        """ Takes a queryVector, relevant documents and non relevant documents
            and returns a new queryVector which is supposedly optimal for better search
        """

        corpus = None
        vocabulary = set()

        beta = 0.75
        gamma = 0.15

        relevantDocs = [None]*len(relevantDocIds)
        nonRelevantDocs = [None]*len(nonRelevantDocIds)

        with open('output.json', 'r') as file:
            corpus = json.load(file)

            for idx, docId in enumerate(relevantDocIds):
                tokens = corpus[docId].split()
                relevantDocs[idx] = tokens
                vocabulary.update(tokens)

            for idx, docId in enumerate(nonRelevantDocIds):
                tokens = corpus[docId].split()
                nonRelevantDocs[idx] = tokens
                vocabulary.update(tokens)

        vocabulary = list(vocabulary)
        # wordToIndex = { word : idx for idx, word in enumerate(vocabulary) }

        idfVector = np.fromiter(
            (
                math.log(self.N/len(self.postingList[word])) for word in vocabulary
            ), dtype=np.float16)

        tfMatrixRelDocs = self._tfMatrix(vocabulary, relevantDocs)
        tfMatrixNonRelDocs = self._tfMatrix(vocabulary, nonRelevantDocs)

        tfIdfMatrixRelDocs = tfMatrixRelDocs*idfVector
        tfIdfMatrixNonRelDocs = tfMatrixNonRelDocs*idfVector

        # Compute Cosine Similarity

        cosineVector = beta * \
            np.mean(tfIdfMatrixRelDocs, axis=0) - gamma * \
            np.mean(tfIdfMatrixNonRelDocs, axis=0)

        cosineVector = np.clip(cosineVector, a_min=0, a_max=None)

        indices = np.argsort(cosineVector)[-terms:0]

        newQueryVector = np.concatenate((queryVector, cosineVector[indices]))

        newTerms = ''
        for idx, term in enumerate(indices):
            if idx + 1 != len(indices):
                newTerms += f'{term} '
            else:
                newTerms += f'{term}'

        newQuery = query + f' newTerms'

        return (newQuery, newQueryVector)

    def _tfMatrix(self, vocabulary, corpus):
        tfMatrix = np.zeros((len(corpus), len(vocabulary)))

        for row in len(corpus):
            tokenCount = Counter(corpus[row])
            tfMatrix[row, :] = np.fromiter((
                1 + math.log(tokenCount[word])
                if tokenCount[word] > 0
                else 0
                for word in vocabulary
            ), dtype=np.float16)

        return tfMatrix


class EvaluationMetrics():
    def __init__(self, file_name) -> None:
        self.relevance_judgements = {}
        dataset = ir_datasets.load("beir/trec-covid")
        self.queries = [query for (_, _, query, _) in dataset.queries_iter()]
        self.se = SearchEngine('index.txt')

        with open(file_name, 'r') as file:
            while (True):
                line = file.readline().rstrip()
                if len(line) == 0:
                    break
                query, doc_id, rel = line.split(',')
                doc_id = int(doc_id)
                rel = int(rel)
                if self.relevance_judgements.get(query):
                    self.relevance_judgements[query].append((doc_id, rel))
                else:
                    self.relevance_judgements[query] = [(doc_id, rel)]

    def mean_average_precision(self):

        mean_avg_prec = 0.0
        total_queries = len(self.queries)

        for idx, query in enumerate(self.queries, start=1):
            document_ranks = self.se.searchQuery(query)
            relevance_list = self.relevance_judgements[query]
            avg_prec = self._average_precision(relevance_list, document_ranks)

            print(f'Precision for query {idx} : {avg_prec}')

            mean_avg_prec += avg_prec

        mean_avg_prec = round(mean_avg_prec/total_queries, 8)

        return mean_avg_prec

    def _average_precision(self, relevanceList, rankedDocuments):
        avg_precision = 0.0
        relevant_docs = 0

        for idx, (docId, _) in enumerate(rankedDocuments):
            relevance = [rel for (doc_id, rel)
                         in relevanceList if doc_id == docId]

            if len(relevance) > 0:
                relevant_docs += 1
                relevance = 0.667 if relevance[0] == 2 else 0.333
            else:
                relevance = 0

            precision_at_k = round(relevant_docs/(idx+1), 8)
            avg_precision += precision_at_k*relevance

        try:
            avg_precision = round(avg_precision/relevant_docs, 8)
        except ZeroDivisionError:
            avg_precision = 0.0

        return avg_precision

if __name__ == '__main__':
    engine = SearchEngine('index.txt')
    res = engine.searchQuery('coronavirus origin')
    engine.display_result(res)
