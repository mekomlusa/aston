import nltk
import pandas as pd
import numpy as np
import os
import sys
import re
from Evaluator import Evaluator

reload(sys)
sys.setdefaultencoding('utf-8')
class Example:
  '''ground truths and words'''
  def __init__(self):
    self.ground_truths = []
    self.sentences = []

class lsa:

  def readTwoFiles(self, txt_file, summary_file):
    contents = []
    f = open(txt_file)
    for line in f:
      contents.append(line)
    f.close()
    summary = '@highlight '
    with open(summary_file, 'r') as f:
      summary += f.read()
    
    contents.append(summary)
    return self.words2Example('\n'.join(contents))  

    # pass

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


  def test(self, e):
    k = 3
    # read files and turn it into a sentences x words matrix
    
    matrix = {}
    
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
    [P_1, R_1, F1_1] = evaluate.ROUGE1(pred = (pred), test = (e.ground_truths) )
    [P_2, R_2, F1_2] = evaluate.ROUGE2(pred = (pred), test = (e.ground_truths) )

    return P_1, R_1, F1_1, P_2, R_2, F1_2
    # print "P: %s" % P
    # print "R: %s" % R
    # print "F-1: %s" % F

    # print "roung2"
    # [P, R, F] = evaluate.ROUGE2(pred = (pred), test = (e.ground_truths) )

    # print "P: %s" % P
    # print "R: %s" % R
    # print "F-1: %s" % F

def main(TEST = True):
  mylsa = lsa()
  # file name
  if TEST:
    sampleDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/cnn_stories/sample/'
    f = sampleDir + '000c835555db62e319854d9f8912061cdca1893e.story'
    print f
    e = self.readFile(f)
    print "number of sentences: %s" % len(e.sentences) 
    
    print mylsa.test(e)
    return

  data_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/cnn_stories/stories/'
  list1 = []
  list2 = []
  cnt = 0
  # cap = 50
  num_sentences = 0
  sum_p_1 = 0
  sum_r_1 = 0
  sum_p_2 = 0
  sum_r_2 = 0

  import csv
  fs = []
  dirname = os.path.dirname(os.path.abspath(__file__))
  filepath = os.path.join(dirname, 'datalist_2K_testing.csv')
  with open(filepath) as f:
    csv_reader = csv.reader(f, delimiter=',')
    num = 0
    for row in csv_reader:
      if num == 0:
        print('column names are %s' % (','.join(row)))
        num += 1
      else:
        name = row[0].split('.')[0] + '.story'
        fs.append(name)
        print(name)
        num+= 1
    print('processed %s files' % num)
  # print(fs)


  for fname in fs:
    
    path = os.path.join(data_path,fname)
    # print(os.path.getsize(path))
    if not os.path.isfile(path):
      continue
    e = mylsa.readFile(path)
    num_sentences += len(e.sentences)
    # print "number of sentences: %s" % len(e.sentences) 
    P_1, R_1, res1, P_2, R_2, res2 = mylsa.test(e)
    # print res
    sum_p_1 += P_1
    sum_p_2 += P_2
    sum_r_1 += R_1
    sum_r_2 += R_2
    list1.append(res1)
    list2.append(res2)
    cnt += 1
    # if cnt > cap:
    #   break
  
  cnt -= 1
  print( 'number of samples: %s' % cnt )
  print('avg number of sentence per sample: %d' % (num_sentences / cnt)  )
  print 'p_ROUGE1: %s' % (sum_p_1 / cnt)  
  print 'r_ROUGE1: %s' % (sum_r_1 / cnt)
  print 'p_ROUGE2: %s' % (sum_p_2 / cnt)  
  print 'r_ROUGE2: %s' % (sum_r_2 / cnt)
  print 'f1_ROUGE1: %s' % (sum(list1) / cnt)  
  print 'f1_ROUGE2: %s' % (sum(list2) / cnt)

if __name__ == "__main__":
  dataDir = os.path.dirname(os.path.abspath(__file__)) + '/cnn_stories/stories/'
  print "data directory is: " + dataDir
  '''
  $ python lsa.py
  data directory is: D:\repositories\638nlp\aston\codes\evaluation/cnn_stories/stories/
  number of samples: 50
  avg number of sentence per sample: 32
  p_ROUGE1: 0.152357150844
  r_ROUGE1: 0.388643769119
  p_ROUGE2: 0.055908642393
  r_ROUGE2: 0.122941125602
  f1_ROUGE1: 0.215715550295
  f1_ROUGE2: 0.075921303082
  '''
  main(False)
  