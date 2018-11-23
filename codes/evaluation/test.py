from __future__ import print_function
import sys
import os
import csv
from lsa import lsa

def test(cap = 2000):
    mylsa = lsa()
    list1 = []
    list2 = []
    cnt = 0
    print("cap is: %s" % cap)
    num_sentences = 0
    sum_p_1 = 0
    sum_r_1 = 0
    sum_p_2 = 0
    sum_r_2 = 0

    dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/'
    print(dirname)
    # fname ="D:\\repositories\\638nlp\\aston\cnn_stories\\test\datalist_2K_testing.csv"
    fname = os.path.join(dirname , 'cnn_stories/datalist_2K_testing.csv')
    print(fname)
    dname = os.path.join( dirname + 'cnn_stories/')
    rows = readCSV(fname)
    for row in rows:
        txt_file = os.path.join(dname + 'text_2K_new/', row[0])
        summary_file = os.path.join(dname , 'summary_2K_new/' , row[1])
        # print('text_file %s' % (txt_file))
        # print('summary_file %s' % (summary_file))
        e = []
        try:
            e = mylsa.readTwoFiles(txt_file, summary_file)
        except:
            print('txt file %s does not exist, go to next' % row[0])
            continue
        num_sentences += len(e.sentences)
        # print(e.sentences)
        if len(e.sentences) == 0:
            continue
        # print ("number of sentences: %s" % len(e.sentences) )
        P_1, R_1, res1, P_2, R_2, res2 = mylsa.test(e)
        # print res
        sum_p_1 += P_1
        sum_p_2 += P_2
        sum_r_1 += R_1
        sum_r_2 += R_2
        list1.append(res1)
        list2.append(res2)
        cnt += 1
        if cnt > cap:
            break
    
    cnt -= 1
    print( 'number of samples: %s' % cnt )
    print('avg number of sentence per sample: %d' % (num_sentences / cnt)  )
    print('p_ROUGE1: %s' % (sum_p_1 / cnt)  )
    print('r_ROUGE1: %s' % (sum_r_1 / cnt))
    print('p_ROUGE2: %s' % (sum_p_2 / cnt)  )
    print('r_ROUGE2: %s' % (sum_r_2 / cnt))
    print('f1_ROUGE1: %s' % (sum(list1) / cnt)  )
    print('f1_ROUGE2: %s' % (sum(list2) / cnt))


def testWithSize():
    mylsa = lsa()
    list1 = []
    list2 = []
    cnt = 0
    cap = 200
    interval = 100
    num_sentences = 0
    sum_p_1 = 0
    sum_r_1 = 0
    sum_p_2 = 0
    sum_r_2 = 0

    dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/'
    print(dirname)
    # fname ="D:\\repositories\\638nlp\\aston\cnn_stories\\test\datalist_2K_testing.csv"
    fname = os.path.join(dirname , 'cnn_stories/datalist_2K_testing.csv')
    print(fname)
    dname = os.path.join( dirname + 'cnn_stories/')
    rows = readCSV(fname)
    for row in rows:
        txt_file = os.path.join(dname + 'text_2K_new/', row[0])
        summary_file = os.path.join(dname , 'summary_2K_new/' , row[1])
        e = []
        try:
            e = mylsa.readTwoFiles(txt_file, summary_file)
        except:
            print('txt file %s does not exist, go to next' % row[0])
            continue
        num_sentences += len(e.sentences)
        if len(e.sentences) == 0:
            continue
        P_1, R_1, res1, P_2, R_2, res2 = mylsa.test(e)
        
        sum_p_1 += P_1
        sum_p_2 += P_2
        sum_r_1 += R_1
        sum_r_2 += R_2
        list1.append(res1)
        list2.append(res2)
        cnt += 1
        if cnt > cap:
            break

        

    cnt -= 1
    print( 'number of samples: %s' % cnt )
    print('avg number of sentence per sample: %d' % (num_sentences / cnt)  )
    print('p_ROUGE1: %s' % (sum_p_1 / cnt)  )
    print('r_ROUGE1: %s' % (sum_r_1 / cnt))
    print('p_ROUGE2: %s' % (sum_p_2 / cnt)  )
    print('r_ROUGE2: %s' % (sum_r_2 / cnt))
    print('f1_ROUGE1: %s' % (sum(list1) / cnt)  )
    print('f1_ROUGE2: %s' % (sum(list2) / cnt))


def readCSV(fname):
    reslist = []
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print('Column names are %s' % ", ".join(row))
                line_count += 1
            else:
                # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                line_count += 1
                reslist.append(row)
        print('Processed lines. %s' % line_count)
    
    return reslist

if __name__ == '__main__':
    args = sys.argv
    # python test.py arg1 arg2 arg3
    # print 'Number of arguments:', len(sys.argv), 'arguments.'
    # print 'Argument List:', str(sys.argv)
    # Number of arguments: 4 arguments.
    # Argument List: ['test.py', 'arg1', 'arg2', 'arg3']
    # getopt.getopt(args, '')
    
    # args0 is the filename
    cap = int(args[1]) if len(args) > 1 else 50
    test(cap)
