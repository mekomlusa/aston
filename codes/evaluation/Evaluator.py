import numpy as np
import re
class Evaluator:

    def Sample(self):
        ground_truth = ["The 15 new cardinals will be installed on February 14", 
                        "They come from countries such as Myanmar and Tonga",
                        "No Americans made the list this time or the previous time in Francis' papacy"]

        predict = ['Pope Francis said Sunday that he would hold a meeting of cardinals on February 14 \"during which I will name 15 new Cardinals who, coming from 13 countries from every continent, manifest the indissoluble links between the Church of Rome and the particular Churches presentin the world,\" according to Vatican Radio.', 
                'Christopher Bellitto, a professor of church history at Kean University in New Jersey, noted that Francis announced his new slate of cardinals on the Catholic Feast of the Epiphany, which commemorates the visit of the Magi to Jesus\' birthplace in Bethlehem.', 
                'Beginning in the 1920s, an increasing number of Latin American churchmen were named cardinals, and in the 1960s, St. John XXIII, whom Francis canonized last year, appointed the first cardinals from Japan, the Philippines and Africa.']
        print "rounge1"
        [P, R, F] = self.Rounge1(pred = predict, test = ground_truth )
        print "P: %s" % P
        print "R: %s" % R
        print "F-1: %s" % F

        print "rounge2"
        [P, R, F] = self.Rounge2(pred = predict, test = ground_truth )
        
        print "P: %s" % P
        print "R: %s" % R
        print "F-1: %s" % F


    def Rounge2(self, test, pred):
        '''
        @params
        test: human made test sentences list of strings
        pred: predicted sentences list of strings
        @return
        p, r, f
        '''
        testWordPairDict, cntTestPair = self.countWordPairs(test)
        predWordPairDict, cntPredPair = self.countWordPairs(pred)

        # compare words in test and pred, if words exist in test and both in pred, we record this word, cnt++, cnts for all times the words appears instead of unique words
        cntFoundPair = 0
        pairsFound = []
        for p in predWordPairDict:
            if p in testWordPairDict:
                testWordPairDict[p] -= 1
                if testWordPairDict[p] == 0:
                    testWordPairDict.pop(p)
                cntFoundPair += 1
                pairsFound.append(p)

        #precision and recall and f1 score, 
        P = float(cntFoundPair) / float(cntPredPair)
        R = float(cntFoundPair) / float(cntTestPair)
        F = 2.0 * (P * R) / (P + R)

        return [P, R, F]

    def countWordPairs(self, sentencesList):
        '''
        @params
        wordList sentences list of strings
        @return
        dictionary for all word pairs appearing in the passage
        '''
        pairs = dict()
        cnt = 0
        #  cnts for word pairs, for start words, marked as (*, word), for end words, marked as (word, #)
        for sentence in sentencesList:
            wordList = re.split('\W+', sentence)
            cnt += 1 + len(wordList)
            for id in range(len(wordList)):
                if id == 0:
                    p = ('*', wordList[0])
                    if not p in pairs:
                        pairs[p] = 0
                    pairs[p] += 1
                else:
                    p = (wordList[id - 1], wordList[id])
                    if not p in pairs:
                        pairs[p] = 0
                    pairs[p] += 1
            p = (wordList[len(wordList) - 1], 'STOP')
            if not p in pairs:
                pairs[p] = 0
            pairs[p] += 1

        return [pairs, cnt]

    def Rounge1(self, test, pred):
        '''
        @params
        test: human made test sentences list of strings
        pred: predicted sentences list of strings
        @return
        P, R, F
        '''
        from collections import Counter
        #cnts for all words in test cases and ground truth cases
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

        # compare words in test and pred, if words exist in test and both in pred, we record this word, cnt++, cnts for all times the words appears instead of unique words
        cnt_extracted = 0
        words_extracted = []
        for word in pred_dict:
            if word in test_dict:
                test_dict[word] -= 1
                if test_dict[word] == 0:
                    test_dict.pop(word)
                cnt_extracted += 1
                words_extracted.append(word)

        #precision and recall and f1 score, 
        P = float(cnt_extracted) / float(len(pred_words))
        R = float(cnt_extracted) / float(len(test_words))
        F = 2.0 * (P * R) / (P + R)

        return [P, R, F]
        # print "P: %s" % P
        # print "R: %s" % R
        # print "F-1: %s" % F

if __name__ == '__main__':
    test = Evaluator()
    test.Sample()