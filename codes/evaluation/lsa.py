import nltk
import pandas as pd
import numpy as np
import os
import sys
import re
from Evaluator import Evaluator
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

    for idx in range(1,len(parts)):
      e.ground_truths.append(parts[idx])    
    return e


  def sample(self, f):
    k = 3
    # read files and turn it into a sentences x words matrix
    e = self.readFile(f)
    matrix = {}
    print "number of sentences: %s" % len(e.sentences) 
    for idx in range(len(e.sentences)):
      matrix[idx] = {} 
      words = re.split('\W+', e.sentences[idx])
      for word in words:
        if  len(word) == 0: # skip ""
          continue
        if word not in matrix[idx]:
          matrix[idx][word] = 0
        matrix[idx][word] += 1
    
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
    top_sentences = saliency_vec.argsort()[-k:][::-1]
    
    top_sentences.sort() # Return the sentences in the order in which they appear in the document

    pred = [e.sentences[i] for i in top_sentences]
    # print pred
    evaluate = Evaluator()
    [P, R, F] = evaluate.rounge1(pred = (pred), test = (e.ground_truths) )

    return F
    # print "P: %s" % P
    # print "R: %s" % R
    # print "F-1: %s" % F

    # print "roung2"
    # [P, R, F] = evaluate.rounge2(pred = (pred), test = (e.ground_truths) )

    # print "P: %s" % P
    # print "R: %s" % R
    # print "F-1: %s" % F

def main():
  # print "main"
  # sampleDir = os.path.dirname(os.path.abspath(__file__)) + '/cnn_stories/sample/'
  mylsa = lsa()

  # file name
  sampleDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/cnn_stories/sample/'
  f = sampleDir + '000c835555db62e319854d9f8912061cdca1893e.story'
  
  print f
    
  print mylsa.sample(f)

  data_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/cnn_stories/stories/'
  res_list = []
  cnt = 0
  for fname in os.listdir(data_path):
    
    path = os.path.join(data_path,fname)
    res = mylsa.sample(path)
    print res
    res_list.append(res)
    cnt += 1
    if cnt > 20:
      break
  
  print 'f1: %s' % (sum(res_list) / len(res_list))

if __name__ == "__main__":
  dataDir = os.path.dirname(os.path.abspath(__file__)) + '/cnn_stories/stories/'
  print "data directory is: " + dataDir
  
  main()
  