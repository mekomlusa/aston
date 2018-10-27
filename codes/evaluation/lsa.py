import nltk
import pandas as pd
import numpy as np
import os
import sys
import re
from Evaluator import evaluator
class Example:
  '''ground truths and words'''
  def __init__(self):
    self.ground_truths = []
    self.sentences = []

class lsa:

  def readFile(self, fileName):
    """
     * Code for reading a file. 
     and turn it into an example
      pass a lot of words splitted into self.words
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    return self.words2Example('\n'.join(contents))  

  def words2Example(self, s):
    """
     * word into 
     >>sentences
     >>ground truthes, 
    """
    parts = s.split("@highlight")
    e = Example()    
    
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(parts[0])
    for line in sentences:
      e.sentences.append(line)

    for id in range(1,len(parts)):
      e.ground_truths.append(parts[id])    
    return e


  def sample(self):
    k = 3
    # file name
    sampleDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/cnn_stories/sample/'
    f = sampleDir + '000c835555db62e319854d9f8912061cdca1893e.story'
    print f
    # read files and turn it into a sentences x words matrix
    e = self.readFile(f)
    matrix = {}
    print "number of sentences: %s" % len(e.sentences) 
    for id in range(len(e.sentences)):
      matrix[id] = {} 
      words = re.split('\W+', e.sentences[id])
      for word in words:
        if  len(word) == 0: # skip ""
          continue
        if word not in matrix[id]:
          matrix[id][word] = 0
        matrix[id][word] += 1
    
    # complete svd here
    arr = pd.DataFrame.from_dict(matrix).fillna(0).values
    # u, s, vh = np.linalg.svd(arr, full_matrices=True) # full matrix
    u, s, vh = np.linalg.svd(arr, full_matrices=False) # reduced matrix
    # print u.shape
    # print s.shape
    # print vh.shape
    # print np.allclose(arr, np.dot(u * s, vh))
    # print vh[:, k]
    threshold = 0.5

    sigma_threshold = max(s) * threshold
    s[s < sigma_threshold] = 0  # Set all other singular values to zero

    saliency_vec = np.dot(np.square(s), np.square(vh)) # Build a "length vector" containing the length (i.e. saliency) of each sentence
    top_sentences = saliency_vec.argsort()[: k][::-1]
    
    top_sentences.sort() # Return the sentences in the order in which they appear in the document

    pred = [e.sentences[i] for i in top_sentences]
    # print pred
    evaluate = evaluator()
    [P, R, F] = evaluate.rounge2(pred = (pred), test = (e.ground_truths) )

    print "P: %s" % P
    print "R: %s" % R
    print "F-1: %s" % F

def main():
  print "main"
  sampleDir = os.path.dirname(os.path.abspath(__file__)) + '/cnn_stories/sample/'
  
  mylsa = lsa()
  mylsa.sample()


if __name__ == "__main__":
  dataDir = os.path.dirname(os.path.abspath(__file__)) + '/cnn_stories/stories/'
  print "data directory is: " + dataDir
  
  main()
  