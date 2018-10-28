# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:41:03 2018

@author: Rose
"""
from __future__ import division
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
import pandas as pd
import numpy as np
from collections import Counter
import h5py
import os
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from nltk.stem import WordNetLemmatizer

text_path = "/root/nlp_project/cnn/text_samples/"
summary_path = "/root/nlp_project/cnn/summary_samples/"
file_list = "/root/nlp_project/cnn/datalist_20K.csv"

# min. input size: 30 (sentences)
MAX_INPUT_SEQ_LENGTH=30
BATCH_SIZE = 20
EPOCHS = 20

# load the data
def data_prep(text_path, summary_path, file_info, save_transformed_flag, transformed_path=None):
    label = []
    feature= []
    data_df = pd.read_csv(file_list,sep=',',header='infer')
    word_tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    
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
            for w in all_docs_words:
                all_docs_words_cleaned.append(lemmatizer.lemmatize(w,pos='v'))
            text_word_counter = Counter(all_docs_words_cleaned)
            all_docs_bigram = list(nltk.bigrams(all_docs_words_cleaned))
            total_num_words = len(all_docs_words_cleaned)
    
        summary_file = os.path.join(summary_path,data_df["summary_path"][i])
        with open(summary_file, 'r', encoding="utf8") as rf:
            text = rf.read()
            summary_array = word_tokenizer.tokenize(text)
            summary_array_cleaned = []
            # add lemmatization here.
            for w in summary_array:
                summary_array_cleaned.append(lemmatizer.lemmatize(w,pos='v'))
        
        # focing the same shape.
        if len(text_sents) > MAX_INPUT_SEQ_LENGTH:
            text_sents = text_sents[:MAX_INPUT_SEQ_LENGTH]
        elif len(text_sents) == 1:
            continue
        else:
            padding_needed = True
            
        # Calculating rouge1 label, as described in the paper.
        for j in range(len(text_sents)):
            both_occur = 0
            cum_prob_word = 0
            cum_prob_bigram = 0
            word_array = word_tokenizer.tokenize(text_sents[j])
            word_array_clean = []
            for w in word_array:
                word_array_clean.append(lemmatizer.lemmatize(w,pos='v'))
            all_sent_bigram = list(nltk.bigrams(word_array_clean))
            for w in word_array_clean:
                if w in summary_array_cleaned:
                    both_occur += 1
                cum_prob_word += text_word_counter[w] / total_num_words
            for bg in all_sent_bigram:
                cum_prob_bigram += all_docs_bigram.count(bg) / len(all_docs_bigram)
            # get the rouge1 score here and append to the vector.
            label_list.append(both_occur/len(summary_array_cleaned))
            # get other X features here.
            is_first_sentence = 0
            if j == 0:
                is_first_sentence = 1
            sent_position = j / len(text_sents)
            sum_basic_score = cum_prob_word / len(text_sents)
            sum_bigram_score = cum_prob_bigram / (len(text_sents) - 1)
            feature_list.append([is_first_sentence, sent_position, sum_basic_score, sum_bigram_score])
            # TODO: add more features here
        
        # for padding (manual)
        if padding_needed:
            for _ in range(len(text_sents),MAX_INPUT_SEQ_LENGTH):
                label_list.append(0)
                feature_list.append([0]*4)
        
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

def save_transformed_data(X, Y, saved_path):
    np.save(os.path.join(saved_path,'features.npy'), X)
    np.save(os.path.join(saved_path,'label.npy'), Y)

def load_transformed_data(X_path, Y_path):
    X = np.load(X_path)
    Y = np.load(Y_path)
    return X, Y

def rankNet(input_size, pretrained_weights=None):
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
    
def inference(x_test, y_test, pretrained_weights):
    model = rankNet(x_test.shape[1:], pretrained_weights=pretrained_weights)
    x_test_reshaped = np.reshape(x_test,(len(x_test),x_test.shape[1],x_test.shape[2]))
    preds = model.predict(x_test_reshaped,verbose=0)
    preds = preds.tolist()
    return preds
    
    
if __name__ == "__main__":
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
                X, Y = data_prep(text_path, summary_path, file_list, True, transformed_path)
            else:
                X, Y = data_prep(text_path, summary_path, file_list, False)
        # splitting the data into training & testing set
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        print("Size of the training set:",X_train.shape)
        print("Size of the testing set:",X_test.shape)
        filepath = input("Where would you like to save the model file?")
        training(X_train, y_train, X_test, y_test, filepath)
    else:
        # inference mode
        print("Making predictions.")
        file_list = input("Please specify the testing file list: ")
        X, Y = data_prep(text_path, summary_path, file_list, False)
        pretrained_weights = input("Where is the saved model weights?")
        predictions = inference(X, Y, pretrained_weights)
        data_df = pd.read_csv(file_list,sep=',',header='infer')
        word_tokenizer = RegexpTokenizer(r'\w+')
        
        for i in range(len(data_df)):
            text_file = text_path+data_df["text_path"][i]
            with open(text_file, 'r', encoding="utf8") as rf:
                text = rf.read()
                text_sents = sent_tokenize(text)
            pred_best_three = sorted(range(len(predictions[i])), key=lambda j: predictions[i][j], reverse=True)[:3]
            generated_summary = " ".join([text_sents[num] for num in pred_best_three])
            print("Generated summary:",generated_summary)
            
            summary_file = summary_path + data_df["summary_path"][i]
            with open(summary_file, 'r', encoding="utf8") as rf:
                text = rf.read()
            print("Actual summary:",text)
            print("")