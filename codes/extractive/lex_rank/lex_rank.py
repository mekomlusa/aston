# encoding: utf-8
from codes.extractive.utilities.calc_freq import calc_idf
from codes.extractive.utilities.constants import MIN_NUM_WORDS_IN_SENT, NUM_STORIES, NUM_SUM_SENTS, WORD_MIN_LEN, \
    ZERO_THRESHOLD
from codes.extractive.utilities.get_ground_truth_sum import get_ground_truth_sum
from codes.evaluation.Evaluator import Evaluator
from heapq import heappush, heappop
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import csv
import io
import math
import numpy as np
import os.path
import re


##

STOP_WORDS = None
DAMPING = 0.85
COS_THRESHOLD = 0.1
TFIDF_THRESHOLD = 2.2
EIGEN_VEC_MAX_ERR = 0.01
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()
USE_SIMILARITY = True


def idf_modified_cosine(s1, s2, idf_dict, stem_dict, choice):
    tf_count_dict1 = dict()
    tf_count_dict2 = dict()
    for w in re.findall(r'[a-zA-Z]+', s1):
        w = w.lower()
        if w in STOP_WORDS:
            continue
        if choice != 0:
            if w in stem_dict:
                w = stem_dict[w]
            else:
                w_stem = STEMMER.stem(w) if choice == 1 else LEMMATIZER.lemmatize(w, pos='v')
                stem_dict[w] = w_stem
                w = w_stem

        if len(w) < WORD_MIN_LEN:
            continue
        tf_count_dict1[w] = tf_count_dict1[w] + 1 if w in tf_count_dict1 else 1
    for w in re.findall(r'[a-zA-Z]+', s2):
        w = w.lower()
        if w in STOP_WORDS:
            continue
        if choice != 0:
            if w in stem_dict:
                w = stem_dict[w]
            else:
                w_stem = STEMMER.stem(w) if choice == 1 else LEMMATIZER.lemmatize(w, pos='v')
                stem_dict[w] = w_stem
                w = w_stem
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


def sentence_similarity(s1, s2, idf_dict, lemma_dict):
    w_count_1 = dict()
    w_count_2 = dict()

    for w in re.findall(r'[a-zA-Z]+', s1):
        w = w.lower()
        if w in STOP_WORDS or len(w) < WORD_MIN_LEN:
            continue
        if w in lemma_dict:
            w = lemma_dict[w]
        else:
            w_lemma = LEMMATIZER.lemmatize(w, pos='v')
            lemma_dict[w] = w_lemma
            w = w_lemma
        w_count_1[w] = w_count_1[w] + 1 if w in w_count_1 else 1

    for w in re.findall(r'[a-zA-Z]+', s2):
        w = w.lower()
        if w in STOP_WORDS or len(w) < WORD_MIN_LEN:
            continue
        if w in lemma_dict:
            w = lemma_dict[w]
        else:
            w_lemma = LEMMATIZER.lemmatize(w, pos='v')
            lemma_dict[w] = w_lemma
            w = w_lemma
        w_count_2[w] = w_count_2[w] + 1 if w in w_count_2 else 1

    # ignore common words
    for w, c in w_count_1.items():
        idf = idf_dict[w] if w in idf_dict else 10.0
        tfidf = c * idf
        if tfidf < TFIDF_THRESHOLD:
            w_count_1.pop(w)

    for w, c in w_count_2.items():
        idf = idf_dict[w] if w in idf_dict else 10.0
        tfidf = c * idf
        if tfidf < TFIDF_THRESHOLD:
            w_count_2.pop(w)

    similarity = 0.0
    total = 0.0
    for w1, c1 in w_count_1.items():
        s1s = wn.synsets(w1)
        if len(s1s) == 0:
            continue
        for w2, c2 in w_count_2.items():
            s2s = wn.synsets(w2)
            if len(s2s) == 0:
                continue
            total += c1 * c2
            cur_sim = wn.path_similarity(s1s[0], s2s[0])
            if cur_sim:
                similarity += c1 * c2 * cur_sim

    return similarity / total if total > 0.0 else 0.0


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
    stem_dict = dict()
    for i in range(0, num_sents):
        for j in range(0, num_sents):
            cos_mat[i][j] = sentence_similarity(sents[i], sents[j], idf_dict, stem_dict) if USE_SIMILARITY else idf_modified_cosine(sents[i], sents[j], idf_dict, stem_dict, choice)
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
                words = re.split('\W+', s)
                if len(words) >= MIN_NUM_WORDS_IN_SENT:
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


def test_evaluation_batch(file_paths, idf_dict, is_damped=False, choice=0):
    count = 0
    ev = Evaluator()
    global_p1 = 0.0
    global_r1 = 0.0
    global_f1 = 0.0
    global_p2 = 0.0
    global_r2 = 0.0
    global_f2 = 0.0
    for story_file_path in file_paths:
        cur_sums = lex_rank_process_article_file(story_file_path, idf_dict, is_damped, choice)
        # print(cur_sums)
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

    num_files = len(file_paths)

    print('rouge 1 results')
    print('Avg. P of {0} samples in rounge 1: {1}'.format(num_files, global_p1 / num_files))
    print('Avg. R of {0} samples in rounge 1: {1}'.format(num_files, global_r1 / num_files))
    print('Avg. F-1 of {0} samples in rounge 1: {1}'.format(num_files, global_f1 / num_files))
    print('rouge 2 results')
    print('Avg. P of {0} samples in rounge 2: {1}'.format(num_files, global_p2 / num_files))
    print('Avg. R of {0} samples in rounge 2: {1}'.format(num_files, global_r2 / num_files))
    print('Avg. F-1 of {0} samples in rounge 2: {1}'.format(num_files, global_f2 / num_files))


def summy_lex_rank_process_article_file(file_path):
    sents = []
    with io.open(file_path, 'r', encoding='utf-8') as article_file:
        for line in article_file:
            if line.find('@highlight') != -1:
                break
            line = line.strip()
            sents.extend(sent_tokenize(line))
    parser = PlaintextParser.from_string(' '.join(sents), Tokenizer('english'))
    summarizer = LexRankSummarizer()
    # Summarize the document with 2 sentences
    sums = summarizer(parser.document, NUM_SUM_SENTS)
    res_list = []
    for summary in sums:
        res_list.append(str(summary))
    return res_list


def test_sumy_lexrank_pack_batch(file_paths):
    count = 0
    ev = Evaluator()
    global_p1 = 0.0
    global_r1 = 0.0
    global_f1 = 0.0
    global_p2 = 0.0
    global_r2 = 0.0
    global_f2 = 0.0
    for story_file_path in file_paths:
        cur_sums = summy_lex_rank_process_article_file(story_file_path)
        # print(cur_sums)
        ground_truth_sums = get_ground_truth_sum(story_file_path)
        # print(ground_truth_sums)
        [p1, r1, f1] = ev.rounge1(cur_sums, ground_truth_sums)
        [p2, r2, f2] = ev.rounge2(cur_sums, ground_truth_sums)
        count += 1
        global_p1 += p1
        global_r1 += r1
        global_f1 += f1
        global_p2 += p2
        global_r2 += r2
        global_f2 += f2

    num_files = len(file_paths)

    print('rouge 1 results')
    print('Avg. P of {0} samples in rounge 1: {1}'.format(num_files, global_p1 / num_files))
    print('Avg. R of {0} samples in rounge 1: {1}'.format(num_files, global_r1 / num_files))
    print('Avg. F-1 of {0} samples in rounge 1: {1}'.format(num_files, global_f1 / num_files))
    print('rouge 2 results')
    print('Avg. P of {0} samples in rounge 2: {1}'.format(num_files, global_p2 / num_files))
    print('Avg. R of {0} samples in rounge 2: {1}'.format(num_files, global_r2 / num_files))
    print('Avg. F-1 of {0} samples in rounge 2: {1}'.format(num_files, global_f2 / num_files))


def evaluate_2k_batch(cur_stories_dir, cur_num_test_files, idf_dict, is_damped, choice):
    names_file_path = os.path.join(cur_stories_dir, 'data/datalist_2K_testing.csv')
    names = []
    count = 0
    with open(names_file_path, 'r') as names_file:
        csv_reader = csv.reader(names_file)
        csv_reader.next()
        for line in csv_reader:
            name = '.'.join(line[0].split('.')[:2])
            path = os.path.join(cur_stories_dir, name)
            # print(path)
            names.append(path)
            count += 1
            if count == cur_num_test_files:
                break

    # print('\nsumy lexrank')
    # test_sumy_lexrank_pack_batch(names)

    print('\ndamped similarity lemma')
    test_evaluation_batch(names, idf_dict, is_damped=is_damped, choice=choice)


if __name__ == '__main__':
    # idf_dict = calc_idf()
    # idf_dict = calc_idf(True)
    idf_dict = calc_idf(choice=2)
    STOP_WORDS = set(stopwords.words('english'))
    dir_name = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(dir_name, os.pardir, os.pardir, os.pardir))
    stories_dir = os.path.join(root_dir, 'cnn_stories')

    num_test_files = 200
    evaluate_2k_batch(stories_dir, num_test_files, idf_dict, is_damped=True, choice=2)



