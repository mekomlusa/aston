import re
import os
import csv
WORD_MIN_LEN = 1


def calc_tf(article_file, words_count_dict):
    with open(article_file, 'r') as article:
        line = article.readline()
        while line:
            if line.find('@highlight') != -1:
                break
            words = re.findall(r'[a-zA-Z]+', line)
            line = article.readline()
            for w in words:
                if len(w) < WORD_MIN_LEN:
                    continue
                w = w.lower()
                words_count_dict[w] = words_count_dict[w] + 1 if w in words_count_dict else 1


# return a dict for document frequencies
def calc_df():
    dirname = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(dirname, os.pardir, os.pardir, os.pardir))
    dict_file = os.path.join(dirname, 'doc_freq', 'df.csv')
    words_count_dict = dict()
    if os.path.isfile(dict_file):
        with open(dict_file, 'r') as csv_file:
            csv_r = csv.reader(csv_file)
            for r in csv_r:
                words_count_dict[r[0]] = int(r[1])
    else:
        stories_dir = os.path.join(root_dir, 'cnn_stories')
        i = 0
        for doc in os.listdir(stories_dir):
            calc_tf(os.path.join(stories_dir, doc), words_count_dict)
            print i
            i += 1

        with open(dict_file, 'w') as csv_file:
            csv_w = csv.writer(csv_file)
            for w, c in words_count_dict.iteritems():
                csv_w.writerow([w, c])
    return words_count_dict


if __name__ == '__main__':
    calc_df()