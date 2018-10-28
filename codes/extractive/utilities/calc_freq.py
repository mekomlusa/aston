from constants import NUM_STORIES, WORD_MIN_LEN
import csv
import math
import os
import re


def calc_tf(article_file, words_count_dict, is_binary=False):
    uniq_words_set = set()
    with open(article_file, 'r') as article:
        for line in article:
            if line.find('@highlight') != -1:
                break
            line = line.strip()
            # skip subtitles
            if len(line) == 0 or line[-1].isalnum():
                continue
            words = re.findall(r'[a-zA-Z]+', line)
            for w in words:
                if len(w) < WORD_MIN_LEN:
                    continue
                w = w.lower()
                if is_binary:
                    if w in uniq_words_set:
                        continue
                    uniq_words_set.add(w)
                words_count_dict[w] = words_count_dict[w] + 1 if w in words_count_dict else 1


# return a dict for document frequencies
def calc_idf(is_binary=False):
    dir_name = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(dir_name, os.pardir, os.pardir, os.pardir))
    extractive_dir = os.path.abspath(os.path.join(dir_name, os.pardir))
    dict_file_name = 'idf_binary.csv' if is_binary else 'idf.csv'
    dict_file_path = os.path.join(extractive_dir, 'utilities', dict_file_name)
    words_count_dict = dict()
    if os.path.isfile(dict_file_path):
        with open(dict_file_path, 'r') as csv_file:
            csv_r = csv.reader(csv_file)
            for r in csv_r:
                words_count_dict[r[0]] = float(r[1])
    else:
        stories_dir = os.path.join(root_dir, 'cnn_stories')
        i = 0
        for doc in os.listdir(stories_dir):
            calc_tf(os.path.join(stories_dir, doc), words_count_dict, is_binary)
            print i
            i += 1

        with open(dict_file_path, 'w') as csv_file:
            csv_w = csv.writer(csv_file)
            for w, c in words_count_dict.iteritems():
                csv_w.writerow([w, math.log(NUM_STORIES * 1.0 / c)])
    return words_count_dict


if __name__ == '__main__':
    calc_idf(True)