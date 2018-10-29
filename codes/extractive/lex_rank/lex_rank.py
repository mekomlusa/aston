# encoding: utf-8
from codes.extractive.utilities.calc_freq import calc_idf
from codes.extractive.utilities.constants import MIN_NUM_WORDS_IN_SENT, NUM_STORIES, NUM_SUM_SENTS, WORD_MIN_LEN, \
    ZERO_THRESHOLD
from codes.extractive.utilities.get_ground_truth_sum import get_ground_truth_sum
from codes.evaluation.Evaluator import Evaluator
from heapq import heappush, heappop
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import io
import math
import numpy as np
import os.path
import re


STOP_WORDS = None
DAMPING = 0.5
COS_THRESHOLD = 0.1
EIGEN_VEC_MAX_ERR = 0.05
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()


def idf_modified_cosine(s1, s2, idf_dict, choice):
    tf_count_dict1 = dict()
    tf_count_dict2 = dict()
    for w in re.findall(r'[a-zA-Z]+', s1):
        if w in STOP_WORDS:
            continue
        if choice == 1:
            w = STEMMER.stem(w)
        elif choice == 2:
            w = LEMMATIZER.lemmatize(w, pos='v')
        if len(w) < WORD_MIN_LEN:
            continue
        tf_count_dict1[w] = tf_count_dict1[w] + 1 if w in tf_count_dict1 else 1
    for w in re.findall(r'[a-zA-Z]+', s2):
        if w in STOP_WORDS:
            continue
        if choice == 1:
            w = STEMMER.stem(w)
        elif choice == 2:
            w = LEMMATIZER.lemmatize(w, pos='v')
        if len(w) < WORD_MIN_LEN:
            continue
        tf_count_dict2[w] = tf_count_dict2[w] + 1 if w in tf_count_dict2 else 1

    res = 0
    tf_idf_sum1 = 0
    for w in tf_count_dict1:
        idf = idf_dict[w] if w in idf_dict else math.log(NUM_STORIES / 2.0)
        tf_idf_sum1 += tf_count_dict1[w] * tf_count_dict1[w] * idf * idf
        if w not in tf_count_dict2:
            continue

        res += tf_count_dict1[w] * tf_count_dict2[w] * idf * idf
    if res < ZERO_THRESHOLD:
        return 0.0

    tf_idf_sum2 = 0
    for w in tf_count_dict2:
        idf = idf_dict[w] if w in idf_dict else math.log(NUM_STORIES / 2.0)
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


def calc_lex_rank_scores(sents, idf_dict, is_damped=False, choice=0):
    num_sents = len(sents)
    cos_mat = np.zeros((num_sents, num_sents))
    deg = np.zeros((num_sents, 1))

    for i in range(0, num_sents):
        for j in range(0, num_sents):
            cos_mat[i][j] = idf_modified_cosine(sents[i], sents[j], idf_dict, choice)
            if cos_mat[i][j] > COS_THRESHOLD:
                cos_mat[i][j] = 1.0
                deg[i] += 1
            else:
                cos_mat[i][j] = 0.0

    for i in range(0, num_sents):
        if deg[i] == 0:
            continue
        for j in range(0, num_sents):
            cos_mat[i][j] /= deg[i]

    if is_damped:
        cos_mat = DAMPING * np.ones((num_sents, num_sents)) / num_sents + (1 - DAMPING) * cos_mat

    return power_method(cos_mat, num_sents, EIGEN_VEC_MAX_ERR)


def lex_rank_process_article_file(file_path, idf_dict, is_damped=False, choice=0):
    sents = []
    with io.open(file_path, 'r', encoding='utf-8') as article_file:
        for line in article_file:
            if line.find('@highlight') != -1:
                break
            line = line.strip()
            # skip subtitles
            if len(line) == 0 or line[-1].isalnum():
                continue
            cur_sents = sent_tokenize(line)
            for s in cur_sents:
                if len(s) >= MIN_NUM_WORDS_IN_SENT:
                    sents.append(s)
    scores = calc_lex_rank_scores(sents, idf_dict, is_damped, choice)
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


def test_evaluation(story_path, idf_dict):
    ground_truth_sums = get_ground_truth_sum(story_path)
    cur_sums = lex_rank_process_article_file(story_path, idf_dict)
    ev = Evaluator()
    ev.print_rouge_1_2(ground_truth_sums, cur_sums)


def test_evaluation_batch(stories_dir_path, num_files, idf_dict, is_damped=False, choice=0):
    count = 0
    ev = Evaluator()
    global_p1 = 0.0
    global_r1 = 0.0
    global_f1 = 0.0
    global_p2 = 0.0
    global_r2 = 0.0
    global_f2 = 0.0
    for story_file in os.listdir(stories_dir_path):
        story_file_path = os.path.join(stories_dir_path, story_file)
        cur_sums = lex_rank_process_article_file(story_file_path, idf_dict, is_damped, choice)
        ground_truth_sums = get_ground_truth_sum(story_file_path)
        [p1, r1, f1] = ev.rounge1(cur_sums, ground_truth_sums)
        [p2, r2, f2] = ev.rounge2(cur_sums, ground_truth_sums)
        count += 1
        global_p1 += p1
        global_r1 += r1
        global_f1 += f1
        global_p2 += p2
        global_r2 += r2
        global_f2 += f2
        if count == num_files:
            break

    print('rouge 1 results')
    print('Avg. P of {0} samples in rounge 1: {1}'.format(num_files, global_p1 / num_files))
    print('Avg. R of {0} samples in rounge 1: {1}'.format(num_files, global_r1 / num_files))
    print('Avg. F-1 of {0} samples in rounge 1: {1}'.format(num_files, global_f1 / num_files))
    print('rouge 2 results')
    print('Avg. P of {0} samples in rounge 2: {1}'.format(num_files, global_p2 / num_files))
    print('Avg. R of {0} samples in rounge 2: {1}'.format(num_files, global_r2 / num_files))
    print('Avg. F-1 of {0} samples in rounge 2: {1}'.format(num_files, global_f2 / num_files))


if __name__ == '__main__':
    idf_dict_original = calc_idf()
    idf_dict_original_bi = calc_idf(True)
    idf_dict_stem = calc_idf(choice=1)
    idf_dict_lemma = calc_idf(choice=2)
    STOP_WORDS = set(stopwords.words('english'))
    dir_name = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(dir_name, os.pardir, os.pardir, os.pardir))
    stories_dir = os.path.join(root_dir, 'cnn_stories')
    # story_file_path = os.path.join(stories_dir, '0a0adc84ccbf9414613e145a3795dccc4828ddd4.story')
    # test_evaluation(story_file_path)
    num_test_files = 200
    print('original')
    test_evaluation_batch(stories_dir, num_test_files, idf_dict_original)
    print('\ndamped')
    test_evaluation_batch(stories_dir, num_test_files, idf_dict_original, is_damped=True)
    print('\nbinary')
    test_evaluation_batch(stories_dir, num_test_files, idf_dict_original_bi)
    # print('\nstemming')
    # test_evaluation_batch(stories_dir, num_test_files, idf_dict_stem, choice=1)
    # print('\nlemma')
    # test_evaluation_batch(stories_dir, num_test_files, idf_dict_lemma, choice=2)


