import nltk
import pandas as pd
import numpy as np
import os
import sys
import re
 
class Example:
  '''ground truths and words'''
  def __init__(self):
    self.ground_truths = []
    self.sentences = []

class lsa:

  # def __init__(self):
    # TODO: 
  
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

  def filterStopWords(self):    
    from nltk.corpus import stopwords
    sw = set(stopwords.words('english'))
    self.wordsFiltered = []
    for word in self.words:
      if not word in sw:
        self.wordsFiltered.add(word)
  

  def sample(self):
    sampleDir = os.path.dirname(os.path.abspath(__file__)) + '/cnn/sample/'
    f = sampleDir + '000c835555db62e319854d9f8912061cdca1893e.story'
    e = self.readFile(f)
    # print e.ground_truths
    # print ' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #'
    # for id in range(len(e.sentences)):
    #   print id 
    #   print e.sentences[id]
    # print ' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #'
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
    # print ' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #'
    k = 1
    arr = pd.DataFrame.from_dict(matrix).fillna(0).values
    print arr
    # u, s, vh = np.linalg.svd(arr, full_matrices=True) # full matrix
    u, s, vh = np.linalg.svd(arr, full_matrices=False) # reduced matrix

    print u.shape
    print s.shape
    print vh.shape

    print np.allclose(arr, np.dot(u * s, vh))

    print vh[:, k]
    threshold = 0.5

    sigma_threshold = max(s) * threshold
    s[s < sigma_threshold] = 0  # Set all other singular values to zero

    # Build a "length vector" containing the length (i.e. saliency) of each sentence
    saliency_vec = np.dot(np.square(s), np.square(vh))

    top_sentences = saliency_vec.argsort()[-3:][::-1]
    # Return the sentences in the order in which they appear in the document
    top_sentences.sort()

    print [e.sentences[i] for i in top_sentences]

    print e.ground_truths

    return [e.sentences[i] for i in top_sentences]

    # arr[id] *  max
    # for id in range(17):
    #   arr[id] * vh[:, k]
    # print matrix

    
    # print ' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #'

def main():
  print "main"
  sampleDir = os.path.dirname(os.path.abspath(__file__)) + '/cnn/sample/'
  print "sample directory is: " + sampleDir
  mylsa = lsa()
  print mylsa.sample()


if __name__ == "__main__":
  dataDir = os.path.dirname(os.path.abspath(__file__)) + '/cnn/stories/'
  print "data directory is: " + dataDir
  
  main()

  # FileNames = os.listdir('%s' % os.path.dirname(os.path.abspath(__file__)))
  # print FileNames
  