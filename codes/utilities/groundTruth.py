# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 10:50:50 2018

@author: Rose Lin

This script can batch process story files and extracting "gold standard" summaries
(collections of @highlight).

Sample usage:
    
    python groundTruth.py -p "D:\Google Drive\CSCE 638\project\samples"
    // Extracting stories located under D:\Google Drive\CSCE 638\project\samples
    and save the summaries and the texts in the same directory.
    
    python groundTruth.py -p "D:\Google Drive\CSCE 638\project\samples" -o "D:\Google Drive\CSCE 638\project\summaries"
    // Extracting stories located under D:\Google Drive\CSCE 638\project\samples
    and save the summaries in D:\Google Drive\CSCE 638\project\summaries
    
Python 3 rewrite
"""

import re
import os
import argparse

def summaryExtractor(storyLocation, outputLocation=None, textLocation=None):
    rawSamples = os.listdir('%s' % storyLocation)
    if storyLocation[:-1] != '/':
        storyLocation += '/'
    
    for sample in rawSamples:
        with open(storyLocation+sample, 'r') as rf:
            data = rf.read()
            # first count how many @highlights
            sentenceCount = len(re.findall(r'@highlight',data)) * -1
            # get the summary and the original text
            text = data.split('@highlight')[:sentenceCount][0]
            text = text.replace('\n',' ')
            sumCollect = data.split('@highlight')[sentenceCount:]
            summary =  ". ".join([s.strip('\n') for s in sumCollect])
            summary += "."
            
        # write out
        sampleOutput = sample+".summary.txt"
        textOutput = sample+".text.txt"
        if not outputLocation and textLocation:
            with open(storyLocation+sampleOutput, 'w') as wf:
                wf.write(summary)
            with open(storyLocation+textOutput, 'w') as wf:
                wf.write(text)
            
        else:
            if outputLocation[-1] != '/':
                outputLocation += '/'
            if textLocation[-1] != '/':
                textLocation += '/'
        
            with open(outputLocation+sampleOutput, 'w') as wf:
                wf.write(summary)
            with open(textLocation+textOutput, 'w') as wf:
                wf.write(text)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='DeepMind CNN News Summary Extraction Tool')
    parser.add_argument('-p','--path', required=True,
                        metavar="/path/to/dataset/",
                        help='Root directory of the stories')
    parser.add_argument('-o','--output', required=False,
                        metavar="/path/to/dataset/",
                        help='Output directory to save the summaries')
    parser.add_argument('-t','--text', required=False,
                        metavar="/path/to/dataset/",
                        help='Output directory to save the text (no summary)')
    args = parser.parse_args()
    
    print("Extracting summaries.")
    if args.output and args.text:
        summaryExtractor(args.path, args.output, args.text)
        print("Summaries could be found under",args.output)
        print("Cleaned text could be found under",args.text)
    else:
        summaryExtractor(args.path)
        print("Summaries could be found under",args.path)
        print("Cleaned text could be found under",args.path)