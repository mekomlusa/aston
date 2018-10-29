import re


class Evaluator:

    def sample(self):
        ground_truth = ["The 15 new cardinals will be installed on February 14", 
                        "They come from countries such as Myanmar and Tonga",
                        "No Americans made the list this time or the previous time in Francis' papacy"]

        predict = ['Pope Francis said Sunday that he would hold a meeting of cardinals on February 14 \"during which I will name 15 new Cardinals who, coming from 13 countries from every continent, manifest the indissoluble links between the Church of Rome and the particular Churches presentin the world,\" according to Vatican Radio.', 
                'Christopher Bellitto, a professor of church history at Kean University in New Jersey, noted that Francis announced his new slate of cardinals on the Catholic Feast of the Epiphany, which commemorates the visit of the Magi to Jesus\' birthplace in Bethlehem.', 
                'Beginning in the 1920s, an increasing number of Latin American churchmen were named cardinals, and in the 1960s, St. John XXIII, whom Francis canonized last year, appointed the first cardinals from Japan, the Philippines and Africa.']
        self.print_rouge_1_2(predict, ground_truth)

    def print_rouge_1_2(self, pred, test):
        print "rounge1"
        [p, r, f] = self.rounge1(pred=pred, test=test)
        print("P: %s" % p)
        print("R: %s" % r)
        print("F-1: %s" % f)

        print "rounge2"
        [p, r, f] = self.rounge2(pred=pred, test=test)

        print("P: %s" % p)
        print("R: %s" % r)
        print("F-1: %s" % f)

    def rounge2(self, test, pred):
        '''
        @params
        test: human made test sentences list of strings
        pred: predicted sentences list of strings
        @return
        p, r, f
        '''
        test_word_pair_dict, cnt_test_pair = self.count_word_pairs(test)
        pred_word_pair_dict, cnt_pred_pair = self.count_word_pairs(pred)

        # compare words in test and pred, if words exist in test and both in pred, we record this word, cnt++, cnts for
        # all times the words appears instead of unique words
        cnt_found_pair = 0
        for p in pred_word_pair_dict:
            if p in test_word_pair_dict:
                test_word_pair_dict[p] -= 1
                if test_word_pair_dict[p] == 0:
                    test_word_pair_dict.pop(p)
                cnt_found_pair += 1

        # precision and recall and f1 score,
        P = float(cnt_found_pair) / float(cnt_pred_pair) if cnt_pred_pair != 0 else 0.0
        R = float(cnt_found_pair) / float(cnt_test_pair) if cnt_test_pair != 0 else 0.0
        F = 2.0 * (P * R) / (P + R) if cnt_found_pair != 0 else 0.0

        return [P, R, F]

    def count_word_pairs(self, sentences_list):
        '''
        @params
        word_list sentences list of strings
        @return
        dictionary for all word pairs appearing in the passage
        '''
        pairs = dict()
        cnt = 0
        # cnts for word pairs, for start words, marked as (*, word), for end words, marked as (word, #)
        for sentence in sentences_list:
            word_list = re.split('\W+', sentence)
            cnt += 1 + len(word_list)
            for id in range(len(word_list)):
                if id == 0:
                    p = ('*', word_list[0])
                    if not p in pairs:
                        pairs[p] = 0
                    pairs[p] += 1
                else:
                    p = (word_list[id - 1], word_list[id])
                    if not p in pairs:
                        pairs[p] = 0
                    pairs[p] += 1
            p = (word_list[len(word_list) - 1], 'STOP')
            if not p in pairs:
                pairs[p] = 0
            pairs[p] += 1

        return [pairs, cnt]

    def rounge1(self, test, pred):
        '''
        @params
        test: human made test sentences list of strings
        pred: predicted sentences list of strings
        @return
        P, R, F
        '''
        from collections import Counter
        # cnts for all words in test cases and ground truth cases
        test_words = re.split('\W+', " ".join(test))
        pred_words = re.split('\W+', " ".join(pred))
        cnts_test = Counter()
        for word in test_words:
            cnts_test[word] += 1
        test_dict = dict(cnts_test)

        cnts_pred = Counter()
        for word in pred_words:
            cnts_pred[word] += 1
        pred_dict = dict(cnts_pred)

        # compare words in test and pred, if words exist in test and both in pred, we record this word, cnt++, cnts for
        # all times the words appears instead of unique words
        cnt_extracted = 0
        for word in pred_dict:
            if word in test_dict:
                test_dict[word] -= 1
                if test_dict[word] == 0:
                    test_dict.pop(word)
                cnt_extracted += 1

        # precision and recall and f1 score,
        P = float(cnt_extracted) / float(len(pred_words)) if len(pred_words) != 0 else 0.0
        R = float(cnt_extracted) / float(len(test_words)) if len(test_words) != 0 else 0.0
        F = 2.0 * (P * R) / (P + R) if cnt_extracted != 0 else 0.0

        return [P, R, F]


if __name__ == '__main__':
    test_ev = Evaluator()
    test_ev.sample()