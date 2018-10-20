# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 19:33:44 2018

@author: Rose Lin

PY3

Sample usage:
    
    python sampleGenerator.py -ss ~/nlp_project/cnn/summaries/ -st ~/nlp_project/cnn/texts/ 
    -ds ~/nlp_project/cnn/summary_samples/ -dt ~/nlp_project/cnn/summary_texts/ -n 20000
    // Copy 20,000 random files from folder "summaries" to folder "summary_samples".
"""

import shutil
import random
import argparse
import os
import pandas as pd
import datetime

def sampleGenerator(src_summary, src_text, summary_output, text_output, num_of_samples):
    summaries = os.listdir('%s' % src_summary)
    texts = os.listdir('%s' % src_text)
    res_list = pd.DataFrame(columns=['text_path','summary_path'])
    if num_of_samples > len(summaries) or num_of_samples > len(texts):
        raise ValueError("ERROR: requested samples are more than the number of files. Aborted")
    if src_text[-1] != '/':
        src_text += '/'
    if src_summary[-1] != '/':
        src_summary += '/'
    added = set()
    smaller = min(len(summaries),len(texts))
    
    for x in range(num_of_samples):
        selected = random.randint(0,smaller-1)
        while selected in added:
            selected = random.randint(0,smaller-1)
        summary_to_copy = summaries[selected]
        text_to_copy = summaries[selected].split("summary")[0]+'text.txt'
        while text_to_copy not in texts:
            selected = random.randint(0,smaller-1)
            summary_to_copy = summaries[selected]
            text_to_copy = summaries[selected].split("summary")[0]+'.text.txt'
        shutil.copy2(src_summary+summary_to_copy, summary_output)
        shutil.copy2(src_text+text_to_copy, text_output)
        res_list.loc[x] = [text_to_copy,summary_to_copy]
        added.add(selected)
        
    output_file(summary_output, res_list,"datalist")
        
# output the dataframes!   
def output_file(dir, df, filen):
    # Output the blocking list.
    today = datetime.datetime.today()
    outputname = filen+str(today).replace(':','-')+".csv"
    if dir[-1] != '/':
        dir += '/'
    if len(df) > 0:
        df.to_csv(dir+outputname,sep=',', na_rep=" ", encoding='utf-8', index_label=False, index=False) 
        print("The script has created "+str(len(df))+" samples today.")
        print("A summary spreadsheet has been saved under "+dir+" as "+filen)

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='DeepMind CNN News Summary Sample Tool')
    parser.add_argument('-ss','--sumsrc', required=True,
                        metavar="/path/to/dataset/",
                        help='Source directory to copy summaries from')
    parser.add_argument('-st','--textsrc', required=True,
                        metavar="/path/to/dataset/",
                        help='Source directory to copy texts from')
    parser.add_argument('-ds','--sumdes', required=True,
                        metavar="/path/to/dataset/",
                        help='Destination directory to copy summaries to')
    parser.add_argument('-dt','--textdes', required=True,
                        metavar="/path/to/dataset/",
                        help='Destination directory to copy texts to')
    parser.add_argument('-n','--num', required=True,
                        metavar="100",
                        help='Number of samples desired')
    args = parser.parse_args()
    
    print("Copying files.")
    sampleGenerator(args.sumsrc, args.textsrc, args.sumdes, args.textdes, int(args.num))
    print("A random shuffled sample summaries could be found under",args.sumdes,"; sample texts are saved under",args.textdes)