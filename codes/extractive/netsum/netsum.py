# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:41:03 2018

@author: Rose

Python 3.5, tested and passed.

Usage: activate a Python 3.5 environment. Run `python netsum.py`.
"""
#from __future__ import division
import sys	
sys.path.append('../../../')

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
import pandas as pd
import numpy as np
from collections import Counter
import argparse
import os
import re
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from nltk.stem import WordNetLemmatizer
from codes.evaluation.Evaluator import Evaluator
from nltk.corpus import wordnet as wn
import csv

# Parameter settings
# min. input size: 30 (sentences)
MAX_INPUT_SEQ_LENGTH=30
BATCH_SIZE = 20
EPOCHS = 20
STOP_WORDS = []
WORD_MIN_LEN = 1
IS_LEMMATIZE = False
IDF_THRESHOLD = 10

news_words = {}

def data_prep(text_path, summary_path, news_words_path, is_lemmatize, file_info, save_transformed_flag, transformed_path=None):
    '''
    Load and clean the data for further preparation.
    
    Parameters:
        text_path (str): path to the preprocessed news text folder.
        summary_path (str): path to the preprocessed summary text folder.
        file_info (str): path to the mapping file.
        save_transformed_flag (bool): whether to save the transformed features and labels into .npy files. Coupled with transformed_path.
        transformed_path (str): path to save the transformed features and labels. Cannot be None if save_transformed_flag == True.
        
    Return:
        X (numpy.ndarray): transformed features.
        Y (numpy.ndarray): transformed labels.
    '''
    label = []
    feature= []
    data_df = pd.read_csv(file_list,sep=',',header='infer')
    word_tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    # Read the news word corpus.
    reader = csv.reader(open(news_words_path, 'r'))
    
    for row in reader:
       k, v = row
       news_words[k] = float(v)
    
    
    for i in range(len(data_df)):
        if i % 100 == 0:
            print("Transformed",i,"files.")
        label_list = []
        feature_list = []
        padding_needed = False
        text_file = os.path.join(text_path,data_df["text_path"][i])
        with open(text_file, 'r', encoding="utf8") as rf:
            text = rf.read()
            text_sents = sent_tokenize(text)
            all_docs_words = word_tokenizer.tokenize(text)
            all_docs_words_cleaned = []
            # add lemmatization here.
            if is_lemmatize:
                for w in all_docs_words:
                    all_docs_words_cleaned.append(lemmatizer.lemmatize(w,pos='v'))
            else:
                all_docs_words_cleaned = all_docs_words
            text_word_counter = Counter(all_docs_words_cleaned)
            all_docs_bigram = list(nltk.bigrams(all_docs_words_cleaned))
            total_num_words = len(set(all_docs_words_cleaned))
    
        summary_file = os.path.join(summary_path,data_df["summary_path"][i])
        with open(summary_file, 'r', encoding="utf8") as rf:
            text = rf.read()
            summary_sents = sent_tokenize(text)
            summary_senses = []
            for sent in summary_sents:
                summary_senses.append(get_sent_word_sens(sent))
            summary_array = word_tokenizer.tokenize(text)
            summary_array_cleaned = []
            if is_lemmatize:
                # add lemmatization here.
                for w in summary_array:
                    summary_array_cleaned.append(lemmatizer.lemmatize(w,pos='v'))
            else:
                summary_array_cleaned = summary_array
        
        # focing the same shape.
        if len(text_sents) > MAX_INPUT_SEQ_LENGTH:
            text_sents = text_sents[:MAX_INPUT_SEQ_LENGTH]
        elif len(text_sents) == 1:
            continue
        else:
            padding_needed = True
           
        #print(text_file)
        # Calculating rouge1 label, as described in the paper.
        # Currently 7 features:
        # is_first_sentence: whether this is the first sentence or not. sent_position: sentence position within the text. sum_basic_score:
        # sum basic score (unigram). sum_bigram_score: sum basic score (bigrams). sent_sim_to_summ: score that measures the similarity between
        # the sentence and the summary. sum_idf: sum of the idf score for this sentence. avg_idf: average of the idf score for this sentence.
        for j in range(len(text_sents)):
            both_occur = 0
            cum_prob_word = 0
            cum_prob_bigram = 0
            wordnet_sent_sim_score = 0
            word_array = word_tokenizer.tokenize(text_sents[j])
            is_first_sentence = 0
            if j == 0:
                is_first_sentence = 1
            sent_position = j / len(text_sents)
            if len(word_array) == 0:
                # for sentence like "...", there is no way to calculate sum basic related scores. So just skip them.
                feature_list.append([is_first_sentence, sent_position, 0,0,0,0,0,0])
                # similarly, the label will be 0 (since there will be no common word)
                label_list.append(0)
                continue
            
            sum_idf = 0
            word_array_clean = []
            for w in word_array:
                if is_lemmatize:
                    word_array_clean.append(lemmatizer.lemmatize(w,pos='v'))
                sum_idf += news_words.get(w,0)
            if not is_lemmatize:
                word_array_clean = word_array
            avg_idf = sum_idf / len(word_array)
            all_sent_bigram = list(nltk.bigrams(word_array_clean))
            for w in word_array_clean:
                if w in summary_array_cleaned:
                    both_occur += 1
                cum_prob_word += text_word_counter[w] / total_num_words
                
            for bg in all_sent_bigram:
                cum_prob_bigram += all_docs_bigram.count(bg) / len(all_docs_bigram)
            # get other X features here.
            sum_basic_score = cum_prob_word / len(set(word_array_clean))
            if len(set(word_array_clean)) > 1:
                sum_bigram_score = cum_prob_bigram / (len(set(word_array_clean)) - 1)
            else:
                # this sentence really has no bigram.
                sum_bigram_score = 0
            # TODO: add more features here
            sent_sim_to_summ = (both_occur/len(summary_array_cleaned)) / len(set(word_array_clean))
            this_sent_sense = get_sent_word_sens(text_sents[j])
            for ss in summary_senses:
                wordnet_sent_sim_score += sentence_similarity(ss,this_sent_sense)
            feature_list.append([is_first_sentence, sent_position, sum_basic_score, sum_bigram_score, sent_sim_to_summ, sum_idf, avg_idf, wordnet_sent_sim_score])
            label_list.append(both_occur/len(summary_array_cleaned))
        
        # for padding (manual)
        if padding_needed:
            for _ in range(len(text_sents),MAX_INPUT_SEQ_LENGTH):
                label_list.append(0)
                feature_list.append([0]*8)
        
        label.append(label_list)
        feature.append(feature_list)
            
    # convert to numpy array
    Y = np.array(label)
    X = np.array(feature)
    
    if save_transformed_flag:
        if transformed_path:
            save_transformed_data(X, Y, transformed_path)
            print("Transformed data has been saved under",transformed_path)
        else:
            # save transformed data in the working directory
            save_transformed_data(X, Y, os.getcwd())
            print("Transformed data has been saved under",os.getcwd())
            
    return X, Y

def get_sent_word_sens(sentence, with_idf_focus = False):
    '''
    Get senses for each valid word in a sentence.
    
    Parameter:
        sentence (str): a string containing a collection of words.
        with_idf_focus (bool): a boolean indicating whether to filter out words based on its IDF score.
        
    Return:
        wordSense (list): a list of extracted sense (default to be the first one)
    '''
    wordSense = []
    for w in re.findall(r'[a-zA-Z]+', sentence):
        if w in STOP_WORDS or len(w) < WORD_MIN_LEN:
            continue
        # skip words that are deemed to be unimportant by the IDF score.
        if with_idf_focus and news_words.get(w) < IDF_THRESHOLD:
            continue
        senses = wn.synsets(w)
        if len(senses) == 0:
            continue
        wordSense.append(senses[0])
    return wordSense

def sentence_similarity(wordSense1, wordSense2):
    '''
    Calculating sentence similarity measurement.
    
    Parameters:
        wordSense1 (list): a list of extracted sense for the first sentence.
        wordSense2 (list): a list of extracted sense for the second sentence.
        
    Return:
        the similarity score (float).
    '''
    similarity = 0.0
    total = 0.0
    if len(wordSense1) == 0 or len(wordSense2) == 0:
        return 0
    
    for sense1 in wordSense1:
        for sense2 in wordSense2:
            total += 1.0
            cur_sim = wn.path_similarity(sense1, sense2)
            if cur_sim:
                similarity += cur_sim

    return similarity / total

def save_transformed_data(X, Y, saved_path):
    '''
    Save the transformed data, X and Y, to the disk.
    
    Parameters:
        X (numpy.ndarray): transformed features from data_prep.
        Y (numpy.ndarray): transformed labels from data_prep.
        saved_path (str): path to save the transformed features and labels.
    '''
    np.save(os.path.join(saved_path,'features.npy'), X)
    np.save(os.path.join(saved_path,'label.npy'), Y)

def load_transformed_data(X_path, Y_path):
    '''
    Load the transformed data from the disk.
    
    Parameters:
        X_path (str): path to the saved transformed features.
        Y_path (str): path to the saved transformed labels.
    
    Return:
        X (numpy.ndarray): transformed features read from file.
        Y (numpy.ndarray): transformed labels read from file.
    '''
    X = np.load(X_path)
    Y = np.load(Y_path)
    return X, Y

def rankNet(input_size, pretrained_weights=None):
    """
    RankNet algorithm, as proposed in https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/emnlp_svore07.pdf.
    
    Parameters:
        input_size (list): size of the expected input tensor.
        pretrained_weights (str): path to the pretrained weight file. Not required at the training time, but is needed at the inference time.
    
    Return:
        model (keras.models): a Keras model object.
    """
    inputs = Input(shape=input_size)
    flat1 = Flatten()(inputs)
    dense1 = Dense(128, activation='relu')(flat1)
    dense2 = Dense(64, activation='relu')(dense1)
    dense3 = Dense(32, activation='relu')(dense2)
    dense4 = Dense(MAX_INPUT_SEQ_LENGTH, activation='softmax')(dense3)
    
    model = Model(input = inputs, output = dense4)
    
    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    print(model.summary())
    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
    
def training(x_train, y_train, x_test, y_test, filepath, pretrained_weights=None):
    """
    Helper method to train a RankNet model.
    
    Parameters:
        x_train (numpy.ndarray): training set of the features.
        y_train (numpy.ndarray): training set of the labels.
        x_test (numpy.ndarray): testing set of the features.
        y_test (numpy.ndarray): testing set of the labels.
        filepath (str): path to save related information of the training data, i.e. checkpoint and the final model weight file.
        pretrained_weights (str): path to the pretrained weight file. Helpful when the training stops unexpectedly, and one would like to
        resume training.
    """
    model = rankNet(x_train.shape[1:], pretrained_weights=pretrained_weights)
    
    # callback information
    checkpoint = ModelCheckpoint(os.path.join(filepath,"ranknet_best_weights.h5"), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopper = EarlyStopping(patience=5)
    
    model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=(x_test, y_test),
                  callbacks=[early_stopper, checkpoint],
                  shuffle=True)
    model.save_weights(os.path.join(filepath, 'rankNet.h5'))
    
def inference(x_test, pretrained_weights):
    """
    Helper method to load a RankNet model and run predictions.
    
    Parameters:
        x_test (numpy.ndarray): testing set of the features.
        pretrained_weights (str): path to the pretrained weight file. Required to initiate a RankNet model.
        
    Return:
        preds (list of list): predictions for all the testing files. Each file will have a prediction of size 30 x 1, indicating the probability
        of each sentence being extracted as a summary sentence.
    """
    model = rankNet(x_test.shape[1:], pretrained_weights=pretrained_weights)
    x_test_reshaped = np.reshape(x_test,(len(x_test),x_test.shape[1],x_test.shape[2]))
    preds = model.predict(x_test_reshaped,verbose=0)
    preds = preds.tolist()
    return preds

def test_evaluation_batch(num_files, all_pred_summary, all_actual_summary):
    """
    Helper method to get the evaluation metric results (i.e. Rouge1 and 2).
    
    Parameters:
        num_files (int): Number of files in a batch.
        all_pred_summary (list of list): list of predicted summaries, with each element containing a series of summary sentences.
        all_actual_summary (list of list): list of actual summaries, with each element containing a series of summary sentences.
    """
    ev = Evaluator()
    global_p1 = 0.0
    global_r1 = 0.0
    global_f1 = 0.0
    global_p2 = 0.0
    global_r2 = 0.0
    global_f2 = 0.0
    
    for i in range(num_files):
        [p1, r1, f1] = ev.rounge1(all_pred_summary[i], all_actual_summary[i])
        [p2, r2, f2] = ev.rounge2(all_pred_summary[i], all_actual_summary[i])
        global_p1 += p1
        global_r1 += r1
        global_f1 += f1
        global_p2 += p2
        global_r2 += r2
        global_f2 += f2

    print('Rouge 1 results')
    print('Avg. P of {0} samples in rounge 1: {1}'.format(num_files, global_p1 / num_files))
    print('Avg. R of {0} samples in rounge 1: {1}'.format(num_files, global_r1 / num_files))
    print('Avg. F-1 of {0} samples in rounge 1: {1}'.format(num_files, global_f1 / num_files))
    print('Rouge 2 results')
    print('Avg. P of {0} samples in rounge 2: {1}'.format(num_files, global_p2 / num_files))
    print('Avg. R of {0} samples in rounge 2: {1}'.format(num_files, global_r2 / num_files))
    print('Avg. F-1 of {0} samples in rounge 2: {1}'.format(num_files, global_f2 / num_files))
    
# main method
if __name__ == "__main__":
    
    # Take path information at runtime
    parser = argparse.ArgumentParser(
        description='DeepMind CNN News - NetSum algorithm')
    parser.add_argument('-t','--text', required=True,
                        metavar="/path/to/dataset/",
                        help='Path to the extracted text directory')
    parser.add_argument('-s','--summary', required=True,
                        metavar="/path/to/dataset/",
                        help='Path to the extracted summary directory')
    parser.add_argument('-m','--mirror', required=True,
                        metavar="/path/to/datalist_file/",
                        help='Path to the matching data list file (text - summary mirroring)')
    parser.add_argument('-i','--idf', required=False,
                        metavar="/path/to/idf_file/",
                        help='Path to the IDF file')
    args = parser.parse_args()
    text_path = args.text
    if text_path[-1] != '/':
        text_path += '/'
    summary_path = args.summary
    if summary_path[-1] != '/':
        summary_path += '/'
    file_list = args.mirror
    news_words_path = args.idf if args.idf else "/root/nlp_project/cnn/idf.csv"
    
    choice = input("Training or inference? (0 for training, 1 for inference) ")
    while choice != '1' and choice != '0':
        choice = input("Training or inference? (0 for training, 1 for inference) ")
    if choice == '0':
        # training mode
        print("Preparing the data.")
        has_transformed_data = input("Have you saved transformed data before? (Y/N)")
        while has_transformed_data.lower() != 'y' and has_transformed_data.lower() != 'n':
            has_transformed_data = input("Have you saved transformed data before? (Y/N)")
        if has_transformed_data.lower() == 'y':
            saved_path = input("Please provide the path where the transformed X and Y are:")
            X_path = os.path.join(saved_path,'features.h5')
            Y_path = os.path.join(saved_path,'labels.h5')
            X, Y = load_transformed_data(X_path, Y_path)
        else:
            saved_tranformed_data = input("Do you want to save transformed data? (Y/N)")
            while saved_tranformed_data.lower() != 'y' and saved_tranformed_data.lower() != 'n':
                saved_tranformed_data = input("Do you want to save transformed data? (Y/N)")
            if saved_tranformed_data.lower() == 'y':
                transformed_path = input("Please specify the path where you'd like to save your transformed data to:")
                X, Y = data_prep(text_path, summary_path, news_words_path, IS_LEMMATIZE, file_list, True, transformed_path)
            else:
                X, Y = data_prep(text_path, summary_path, news_words_path, IS_LEMMATIZE, file_list, False)
        # splitting the data into training & testing set
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        print("Size of the training set:",X_train.shape)
        print("Size of the testing set:",X_test.shape)
        filepath = input("Where would you like to save the model file?")
        training(X_train, y_train, X_test, y_test, filepath)
    else:
        # inference mode
        all_predicted_summary = []
        all_actual_summary = []
        print("Making predictions.")
        X, Y = data_prep(text_path, summary_path, news_words_path, IS_LEMMATIZE, file_list, False)
        pretrained_weights = input("Where is the saved model weights?")
        predictions = inference(X, pretrained_weights)
        data_df = pd.read_csv(file_list,sep=',',header='infer')
        word_tokenizer = RegexpTokenizer(r'\w+')
        
        for i in range(len(data_df)):
            text_file = text_path+data_df["text_path"][i]
            with open(text_file, 'r', encoding="utf8") as rf:
                text = rf.read()
                text_sents = sent_tokenize(text)
            # handle short articles
            if len(text_sents) < 30:
                for _ in range(len(text_sents),30):
                    text_sents.append("")
            pred_best_three = sorted(range(len(predictions[i])), key=lambda j: predictions[i][j], reverse=True)[:3]
            print("Working on text file:",text_file)
            #print(pred_best_three)
            generated_summary = " ".join([text_sents[num] for num in pred_best_three if text_sents[num] != ""])
            print("Generated summary:",generated_summary)
            all_predicted_summary.append(sent_tokenize(generated_summary))
            
            summary_file = summary_path + data_df["summary_path"][i]
            with open(summary_file, 'r', encoding="utf8") as rf:
                text = rf.read()
            print("Actual summary:",text)
            print("")
            all_actual_summary.append(sent_tokenize(text))
            
        # Rogue 1 and 2 evaluations
        test_evaluation_batch(50, all_predicted_summary, all_actual_summary)