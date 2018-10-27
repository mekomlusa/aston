from codes.extractive.utilities.calc_freq import calc_idf
from codes.extractive.utilities.constants import MIN_NUM_WORDS_IN_SENT, NUM_STORIES, NUM_SUM_SENTS, WORD_MIN_LEN, \
    ZERO_THRESHOLD
from heapq import heappush, heappop
from nltk.corpus import stopwords
import math
import numpy as np
import os.path
import re


IDF_DICT = None
STOP_WORDS = None
DAMPING = 0.85
COS_THRESHOLD = 0.1
EIGEN_VEC_MAX_ERR = 0.05


def idf_modified_cosine(s1, s2):
    tf_count_dict1 = dict()
    tf_count_dict2 = dict()
    for w in re.findall(r'[a-zA-Z]+', s1):
        if w in STOP_WORDS or len(w) < WORD_MIN_LEN:
            continue
        tf_count_dict1[w] = tf_count_dict1[w] + 1 if w in tf_count_dict1 else 1
    for w in re.findall(r'[a-zA-Z]+', s2):
        if w in STOP_WORDS or len(w) < WORD_MIN_LEN:
            continue
        tf_count_dict2[w] = tf_count_dict2[w] + 1 if w in tf_count_dict2 else 1

    res = 0
    tf_idf_sum1 = 0
    for w in tf_count_dict1:
        idf = IDF_DICT[w] if w in IDF_DICT else math.log(NUM_STORIES / 2.0)
        tf_idf_sum1 += tf_count_dict1[w] * tf_count_dict1[w] * idf * idf
        if w not in tf_count_dict2:
            continue

        res += tf_count_dict1[w] * tf_count_dict2[w] * idf * idf
    if res < ZERO_THRESHOLD:
        return 0.0

    tf_idf_sum2 = 0
    for w in tf_count_dict2:
        idf = IDF_DICT[w] if w in IDF_DICT else math.log(NUM_STORIES / 2.0)
        tf_idf_sum2 += tf_count_dict2[w] * tf_count_dict2[w] * idf * idf

    return res * 1.0 / math.sqrt(tf_idf_sum1) * math.sqrt(tf_idf_sum2)


def power_method(cos_mat, num_sents, err):
    p = np.ones((num_sents, 1))
    p /= 1.0 * num_sents
    mat_trans = np.transpose(cos_mat)
    delta = err + 1.0
    while delta > err:
        p1 = np.matmul(mat_trans, p)
        diff = p1 - p
        delta = 0
        for d in diff:
            delta += d * d
        p = p1
    return p


def calc_lex_rank_scores(sents, is_damped=False):
    num_sents = len(sents)
    cos_mat = np.zeros((num_sents, num_sents))
    deg = np.zeros((num_sents, 1))

    for i in range(0, num_sents):
        for j in range(0, num_sents):
            cos_mat[i][j] = idf_modified_cosine(sents[i], sents[j])
            if cos_mat[i][j] > COS_THRESHOLD:
                cos_mat[i][j] = 1
                deg[i] += 1
            else:
                cos_mat[i][j] = 0

    for i in range(0, num_sents):
        for j in range(0, num_sents):
            cos_mat[i][j] /= deg[i]

    if is_damped:
        cos_mat = DAMPING * np.ones((num_sents, num_sents)) / num_sents + (1 - DAMPING) * cos_mat

    return power_method(cos_mat, num_sents, EIGEN_VEC_MAX_ERR)


def process_article_file(file_path, is_damped=False):
    sents = []
    with open(file_path, 'r') as file:
        for line in file:
            if len(line) >= MIN_NUM_WORDS_IN_SENT:
                sents.append(line)
    scores = calc_lex_rank_scores(sents, is_damped)
    min_heap = []
    count = 0
    for i, s in enumerate(scores):
        heappush(min_heap, (s, sents[i]))
        if count < NUM_SUM_SENTS:
            count += 1
        else:
            heappop(min_heap)

    res_list = []
    while len(min_heap) > 0:
        res_list.insert(0, heappop(min_heap)[1])
    return res_list


if __name__ == '__main__':
    IDF_DICT = calc_idf()
    STOP_WORDS = set(stopwords.words('english'))
    dir_name = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(dir_name, os.pardir, os.pardir, os.pardir))
    stories_dir = os.path.join(root_dir, 'cnn_stories')
    for doc in os.listdir(stories_dir):
        sums = process_article_file(os.path.join(stories_dir, doc))
        print(sums)
        break