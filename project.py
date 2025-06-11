# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


_last_request_time = 0

def get_book(url):
    global _last_request_time
    
    robots_url = "https://www.gutenberg.org/robots.txt"
    robots_response = requests.get(robots_url)
    crawl_delay = 0.5
    
    if robots_response.status_code == 200:
        for line in robots_response.text.splitlines():
            if line.lower().startswith("crawl-delay"):
                parts = line.split(":")
                if len(parts) == 2:
                    try:
                        crawl_delay = float(parts[1].strip())
                    except ValueError:
                        pass
    
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < crawl_delay:
        time.sleep(crawl_delay - elapsed)
    
    response = requests.get(url)
    _last_request_time = time.time()
    
    if response.status_code != 200:
        raise Exception(f"Failed to download book: status code {response.status_code}")
    
    text = response.text.replace('\r\n', '\n')
    
    start_match = re.search(r'\*\*\* START OF (.*?) \*\*\*', text)
    end_match = re.search(r'\*\*\* END OF (.*?) \*\*\*', text)
    
    if not start_match or not end_match:
        raise Exception("Start or End markers not found in the book text.")
    
    start_index = start_match.end()
    end_index = end_match.start()
    
    book_contents = text[start_index:end_index]
    
    return book_contents



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    tokens = []
    
    book_string = book_string.replace('\r\n', '\n')
    
    paragraphs = re.split(r'\n\s*\n+', book_string)

    for i in paragraphs:
        i = i.strip()
        if i:  
            tokens.append('\x02')
            para_tokens = re.findall(r'[A-Za-z0-9_]+|[^\s\w]', i)
            tokens.extend(para_tokens)
            tokens.append('\x03')
    
    return tokens



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):
    def __init__(self, corpus=None):
        self.mdl = None
        if corpus is not None:
            self.train(corpus)
    
    def train(self, corpus):
        unique_tokens = sorted(set(corpus))
        prob = 1 / len(unique_tokens)
        self.mdl = pd.Series(data=[prob]*len(unique_tokens), index=unique_tokens)
    
    def probability(self, tokens):
        if self.mdl is None:
            raise Exception("Model has not been trained yet.")
        
        try:
            probs = self.mdl[list(tokens)]
            return probs.prod()
        except KeyError:
            return 0.0
    
    def sample(self, M):
        if self.mdl is None:
            raise Exception("Model has not been trained yet.")
        
        tokens = np.random.choice(self.mdl.index, size=M, replace=True, p=self.mdl.values)
        return ' '.join(tokens)



# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        total = len(tokens)
        counts = pd.Series(tokens).value_counts()
        probs = counts / total
        return probs
    
    def probability(self, words):
        prob = 1.0
        for word in words:
            prob *= self.mdl.get(word, 0)
        return prob
        
    def sample(self, M):
        samples = np.random.choice(
            self.mdl.index,
            size=M,
            p=self.mdl.values
        )
        return ' '.join(samples)



# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        grams = []
        for i in range(len(tokens) +1 - self.N):
            grams.append(tuple(tokens[i: i + self.N]))
        return grams
        
    def train(self, ngrams):
        df = pd.DataFrame(ngrams, columns=[f'w{i}' for i in range(self.N)])
        df['ngram'] = df.apply(lambda row: tuple(row), axis=1)
        df['n1gram'] = df.apply(lambda row: tuple(row)[:-1], axis=1)

        ngram_counts = df['ngram'].value_counts().reset_index()
        ngram_counts.columns = ['ngram', 'count']

        ngram_counts['n1gram'] = ngram_counts['ngram'].apply(lambda x: x[:-1])

        n1gram_counts = ngram_counts.groupby('n1gram')['count'].sum().reset_index()
        n1gram_counts.columns = ['n1gram', 'n1_count']

        merged = pd.merge(ngram_counts, n1gram_counts, on='n1gram')
        merged['prob'] = merged['count'] / merged['n1_count']

        return merged[['ngram', 'n1gram', 'prob']]

    
    def probability(self, words):
        if len(words) < self.N:
            return self.prev_mdl.probability(words)

        prob = 1.0

        if self.N > 2:
            prob *= self.prev_mdl.probability(words[:self.N - 1])
        else:
            prob *= self.prev_mdl.probability((words[0],))

        for i in range(self.N - 1, len(words)):
            ngram = tuple(words[i - self.N + 1: i + 1])
            match = self.mdl[self.mdl['ngram'] == ngram]

            if match.empty:
                return 0.0
            prob *= match.iloc[0]['prob']

        return prob


    

    def sample(self, M):
        output = ['\x02']

        while len(output) - 1 < M:
            context = tuple(output[-(self.N - 1):])
            
            candidates = self.mdl[self.mdl['n1gram'] == context]

            if candidates.empty:
                next_word = '\x03'
            else:
                next_tokens = [ng[-1] for ng in candidates['ngram']]
                probs = candidates['prob'].values
                next_word = np.random.choice(next_tokens, p=probs)

            output.append(next_word)

        return ' '.join(output)
