# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 10:50:50 2018

@author: Rose Lin

This script can batch process story files and extracting "gold standard" summaries
(collections of @highlight).

Sample usage:
    
    python groundTruth.py -p "D:\Google Drive\CSCE 638\project\samples"
    // Extracting stories located under D:\Google Drive\CSCE 638\project\samples
    and save the summaries in the same directory.
    
    python groundTruth.py -p "D:\Google Drive\CSCE 638\project\samples" -o "D:\Google Drive\CSCE 638\project\summaries"
    // Extracting stories located under D:\Google Drive\CSCE 638\project\samples
    and save the summaries in D:\Google Drive\CSCE 638\project\summaries
"""

import re
import os
import argparse

def summaryExtractor(storyLocation, outputLocation=None):
    rawSamples = os.listdir('%s' % storyLocation)
    if storyLocation[:-1] != '\\':
        storyLocation += '\\'
    
    for sample in rawSamples:
        with open(storyLocation+sample, 'r') as rf:
            data = rf.read()
            # first count how many @highlights
            sentenceCount = len(re.findall(r'@highlight',data)) * -1
            # get the summary
            sumCollect = data.split('@highlight')[sentenceCount:]
            summary =  ". ".join([s.strip('\n') for s in sumCollect])
            summary += "."
            
        # write out
        sampleOutput = sample+".summary.txt"
        if not outputLocation:
            with open(storyLocation+sampleOutput, 'wb') as wf:
                wf.write(summary)
        else:
            if outputLocation[:-1] != '\\':
                outputLocation += '\\'
        
            with open(outputLocation+sampleOutput, 'wb') as wf:
                wf.write(summary)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='DeepMind CNN News Summary Extraction Tool')
    parser.add_argument('-p','--path', required=True,
                        metavar="/path/to/dataset/",
                        help='Root directory of the stories')
    parser.add_argument('-o','--output', required=False,
                        metavar="/path/to/dataset/",
                        help='Output directory to save the summaries')
    args = parser.parse_args()
    
    print "Extracting summaries."
    if args.output:
        summaryExtractor(args.path, args.output)
        print "Summaries could be found under",args.output
    else:
        summaryExtractor(args.path)
        print "Summaries could be found under",args.path