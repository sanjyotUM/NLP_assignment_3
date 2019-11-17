#!/usr/bin/env python
# coding: utf-8

from html.parser import HTMLParser
import os
import math
import string
import warnings
import sys

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# nltk.download()

warnings.filterwarnings("ignore")


class BNCParser(HTMLParser):
    # Parses the data in British National Corpus
    def __init__(self):
        HTMLParser.__init__(self)
        self.parsed_instances = list()
        self.recording = False
        self.this_instance = dict()
        self.this_data = list()
        self.ps = PorterStemmer()
        self.stopwords = nltk.corpus.stopwords.words('english')
        
    def handle_starttag(self, tag, attrs):
        if tag == 'instance':
            self.this_instance['id'] = attrs[0][1]  # Id extraction
        if tag == 'context':
            self.recording =  True
    
    def handle_startendtag(self, tag, attrs):
        if tag == 'answer':
            self.this_instance['sense'] = attrs[1][1].split('%')[1]  # Label extraction
            
    def handle_data(self, data):
        if self.recording:
            self.this_data.append(data)
            
    def handle_endtag(self, tag):
        if tag == 'context':
            self.recording = False
        if tag == 'instance':
            self.this_data = ' '.join(self.this_data)
            self.this_data = self.preprocess_text(self.this_data)
            self.this_instance['data'] = self.this_data
            self.parsed_instances.append(self.this_instance)

            self.this_instance = dict()
            self.this_data = list()
            
    def preprocess_text(self, txt):
        txt_without_punc = txt.lower().translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        tokenized_txt = [w for w in word_tokenize(txt_without_punc) if w not in self.stopwords]  # Tokenize
        stemmed_tokens = [self.ps.stem(w) for w in tokenized_txt]  # Stem
        stemmed_tokens = [w for w in stemmed_tokens if not w.isdigit()]  # Remove numeric tokens
        return stemmed_tokens


def get_f_given_s_df(train, vocab_size, disambiguate_word):
    # Get log probabilities of word given sense from training data

    f_given_s = (
        train
        .assign(sense_count = lambda x: x.groupby('sense')['sense'].transform('count'))  # Sense count extraction
        .explode('data')  # Split sentences in corpus to rows with individual words
        .groupby(['data', 'sense', 'sense_count'])
        .agg({'data': 'count'})  # Get word, sense counts
        .rename(columns={'data': 'count'})
        .reset_index()
        .sort_values(by='count', ascending=False)
        .loc[lambda x: x['data'] != disambiguate_word]  # Remove the target word
        .assign(
            smooth_count = lambda x: x['count'] + 1,  # Get add one smoothing counts
            smooth_sense_count = lambda x: x['sense_count'] + vocab_size  # Get add one smoothing counts
        )
        .assign(prob = lambda x: np.log(x['smooth_count']/x['smooth_sense_count']))  # Add one smoothed probability
        .set_index(['data', 'sense'])
        ['prob']
    )
    return f_given_s


def get_f_given_s_prob(f, s, f_given_s, sense_count_dict, vocab_size):
    # Get feature given sense probability and handle for features not present by returning
    # smoothed probability

    try:
        return f_given_s[(f, s)]
    except KeyError:
        return np.log(1.0/(sense_count_dict[s] + vocab_size))


def get_s_prob(s, sense_count_dict):
    # Returns sense probability

    return np.log(sense_count_dict[s]/sum(sense_count_dict.values()))


def get_vocab_size(df):
    # Returns vocab size

    list_of_list = df['data'].values
    return len(set([word for lst in list_of_list for word in lst]))


def predict_sense(word_list, f_given_s, sense_count_dict, vocab_size):
    # Predict sense via Naive Bayes word sense disambiguation

    val = dict()
    for s in sense_count_dict.keys():
        current_val = 0
        for word in word_list:
            current_val += get_f_given_s_prob(word, s, f_given_s, 
                                              sense_count_dict, vocab_size)  # Feature given sense probabilities
        val[s] = current_val + get_s_prob(s, sense_count_dict)  # Sense probability
    return max(val, key=val.get), val


def split_data(df, folds):
    # Split input data into `folds` parts

    n = len(df)
    elem_count = [math.ceil(n/folds)] * (folds - 1) + [n - (folds - 1) * math.ceil(n/folds)]
    
    chunks = []
    for count in elem_count:
        chunks.append(df[:count])
        df = df.iloc[count:, :]
    return chunks


def get_train_test(split_data, test_index):
    # Combine split data into two sets, training and testing

    test = split_data[test_index]
    train_data = [split_data[i] for i in range(len(split_data)) 
                  if i != test_index]
    
    train = pd.DataFrame()
    for data in train_data:
        train = train.append(data)
    return train, test


def accuracy_score(s1, s2):
    # Calculate accuracy

    correct = np.sum(s1 == s2)
    total = len(s1)
    acc = round((correct/total) * 100, 2)
    return correct, total, acc


def write_test_file(test, disambiguate_word, file):
    # Write file in the prescribed format

    test['output'] = disambiguate_word + '%' + test['prediction']
    test = test[['id', 'output']]
    test.to_csv(file, index=False, sep=' ', header=False, mode='a')
    return


if __name__ == '__main__':
    filename = sys.argv[1]
    folds = 5

    disambiguate_word = filename.split('.')[0].split('-')[0]
    output_filename = '{}.wsd.out'.format(disambiguate_word)

    # Parse whole document in a dataframe
    parser = BNCParser()
    with open(filename, 'r') as f:
        parser.feed(f.read())
    df = pd.DataFrame(parser.parsed_instances)

    # Remove output file, if any
    try:
        os.remove(output_filename)
    except FileNotFoundError:
        pass

    # Initialization
    correct = 0
    total = 0
    outdf = pd.DataFrame()

    # Train and predict sense and write output files
    with open(output_filename, 'a') as f:
        for test_fold_index in range(folds):
            train, test = get_train_test(split_data(df, 5), test_fold_index)  # Train test data for this fold iteration

            sense_count_dict = train.sense.value_counts().to_dict()
            vocab_size = get_vocab_size(train)
            f_given_s = get_f_given_s_df(train, vocab_size)

            def predict_func_with_context(
                    word_list,
                    f_given_s=f_given_s,
                    sense_count_dict=sense_count_dict,
                    vocab_size=vocab_size
            ):
                # Construct simpler function for use in dataframe.apply method
                return predict_sense(word_list, f_given_s, sense_count_dict, vocab_size)

            test['func_out'] = test.apply(lambda x: predict_func_with_context(x['data']), axis=1)  # Predict
            test['prediction'] = test['func_out'].str[0]
            test['scores'] = test['func_out'].str[1]

            # Compute accuracy
            fold_correct, fold_total, acc = accuracy_score(test.sense, test.prediction)
            correct += fold_correct
            total += fold_total

            # Write file chunk for this test fold
            f.write('Fold {}\n'.format(test_fold_index+1))
            write_test_file(test, disambiguate_word, f)
            print(test_fold_index, acc)

        print('Average: {}'.format(round((correct/total) * 100, 2)))
