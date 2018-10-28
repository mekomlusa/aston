from codes.extractive.utilities.calc_freq import calc_idf, calc_tf
from codes.extractive.utilities.constants import MIN_NUM_WORDS_IN_SENT, NUM_SUM_SENTS
from heapq import heappush, heappop
from nltk.tokenize import sent_tokenize
import os.path
import re
import time


NUM_TAGS = 20


def calc_sum(article_path, idf_dict):
    tf_dict = dict()
    calc_tf(article_path, tf_dict)

    min_heap = []
    count = 0
    for w in tf_dict:
        tfidf = tf_dict[w] * idf_dict[w]
        heappush(min_heap, (tfidf, w))
        if count < NUM_TAGS:
            count += 1
        else:
            heappop(min_heap)
    tag_tfidf_dict = dict()
    for val in min_heap:
        tag_tfidf_dict[val[1]] = val[0]

    min_sum_heap = []
    count = 0
    with open(article_path, 'r') as article:
        for line in article:
            if line.find('@highlight') != -1:
                break
            line = line.strip()
            # skip subtitles
            if len(line) == 0 or line[-1].isalnum():
                continue
            cur_sents = sent_tokenize(line)
            for sent in cur_sents:
                words = re.findall(r'[a-zA-Z]+', sent)
                if len(words) < MIN_NUM_WORDS_IN_SENT:
                    continue
                score = 0
                for w in words:
                    if w in tag_tfidf_dict:
                        score += tag_tfidf_dict[w]
                heappush(min_sum_heap, (score, sent))
                if count < NUM_SUM_SENTS:
                    count += 1
                else:
                    heappop(min_sum_heap)

    sum_list = []
    while len(min_sum_heap) > 0:
        sum_list.insert(0, heappop(min_sum_heap))

    for tup in sum_list:
        print tup[0]
        print tup[1]


if __name__ == '__main__':
    global_df_dict = calc_idf()
    dir_name = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(dir_name, os.pardir, os.pardir, os.pardir))
    stories_dir = os.path.join(root_dir, 'cnn_stories')
    time_s = time.time()
    # process first article
    for doc in os.listdir(stories_dir):
        calc_sum(os.path.join(stories_dir, doc), global_df_dict)
        break
    print 'process 1 article takes ' + str(time.time() - time_s) + ' seconds'
