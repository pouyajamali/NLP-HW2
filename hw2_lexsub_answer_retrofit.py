import os, sys, optparse, re
import tqdm
import pymagnitude
import numpy as np

class LexSub:

    def __init__(self, wvec_file, topn=10):
        self.wvecs = pymagnitude.Magnitude(wvec_file)
        self.topn = topn

    def substitutes(self, index, sentence):
        "Return ten guesses that are appropriate lexical substitutions for the word at sentence[index]."
        return(list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn))))

    def updater(self, num_iteration):

        #list(map(lambda k: k, self.wvecs.get_vectors_mmap()))

        ''' Read lexicon file and create a dict'''
        lexicon = read_lexicon('ppdb-xl.txt')

        ''' since reading all pymagnitude words is time-consuming,
        you should just read the words in debug_subset. if it worked well,
        continue with reading the whole pymagnitude.
        '''
        debug_subset = get_words_subset(lexicon)

        words = []
        for key, vector in self.wvecs:
            words.append(key)
        wvVocab = set(words) #set it to debug_subset if you need to debug in less time
        loopVocab = wvVocab.intersection(set(lexicon.keys()))


        '''initialiazing Q and Q_hat'''
        Q_hat = {}
        Q = {}
        for w in wvVocab:
            Q_hat[w] = self.wvecs.query(w)
            Q[w] = self.wvecs.query(w)


        for iter in range(num_iteration):
            print("-------------iter:{0}-------------".format(iter))
            # TODO: in every iteration, do the following lines for every word in Q
            for word in tqdm.tqdm(loopVocab):

                ''' this section is not complete yet.'''
                qj_word = set(lexicon[word]).intersection(wvVocab) #lexicon[word] # read from a lexicon to find neighbors
                num_neighbor = len(qj_word)

                # if there isn't any neighbor we don't need to update
                if num_neighbor == 0:
                    continue

                qj = []

                for w in qj_word:
                    qj.append(Q[w])

                    qi_hat = Q_hat[word]

                    alpha_i = 1
                    beta_i_j = 1

                    sigma_qj_beta_i_j = 0
                    for j in range(len(qj)):
                      sigma_qj_beta_i_j = np.add(sigma_qj_beta_i_j, np.multiply(qj[j], beta_i_j))

                    sigma_beta_i_j = 0
                    for j in range(len(qj)):
                      sigma_beta_i_j += beta_i_j
                      Q[word] = np.divide(np.add(sigma_qj_beta_i_j, np.multiply(qi_hat, alpha_i)), sigma_beta_i_j + alpha_i)

        return Q


''' Read the PPDB word relations as a dictionary '''
def read_lexicon(filename):
  lexicon = {}
  for line in open(os.path.join('data', 'lexicons', filename), 'r'):
    words = line.lower().strip().split()
    lexicon[reformat(words[0])] = [reformat(word) for word in words[1:]]
  return lexicon

def write_wvec_to_file(wordVectors, outputFile):
    outFile = open(outputFile, 'w')

    for word, values in wordVectors.items(): #.iteritems():

        outFile.write(word + ' ')
        for val in wordVectors[word]:
            outFile.write('%.5f' % (val) + ' ')
        outFile.write('\n')
    outFile.close()

isNumber = re.compile(r'\d+.*')

# https://github.com/mfaruqui/retrofitting/blob/master/retrofit.py
def reformat(word):
  if isNumber.search(word.lower()):
    return '---num---'
  elif re.sub(r'\W+', '', word) == '':
    return '---punc---'
  else:
    return word.lower()


def get_words_subset(lexicon_words):
    words = []
    syn_words = []
    dev_file = os.path.join('data', 'input', 'dev.txt')
    test_file = os.path.join('data', 'input', 'test.txt')
    '''
    for line in open(dev_file, 'r'):
        word_list = set(map(lambda x: reformat(x), line.lower().strip().split()))
        words.extend(word_list)
    '''
    for line in open(test_file, 'r'):
        word_list = set(map(lambda x: reformat(x), line.lower().strip().split()))
        words.extend(word_list)

    for line in open(test_file, 'r'):
        word_list = set(map(lambda x: reformat(x), line.lower().strip().split()))
        words.extend(word_list)

    for w in words:
        synonyms = lexicon_words.get(w)
        if synonyms != None:
            syn_words.extend(synonyms)
    words.extend(syn_words)
    return set(words)


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-w", "--wordvecfile", dest="wordvecfile", default=os.path.join('data', 'glove.6B.100d.magnitude'), help="word vectors file")
    optparser.add_option("-n", "--topn", dest="topn", default=10, help="produce these many guesses")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)


    lexsub = LexSub(opts.wordvecfile, int(opts.topn))
    num_lines = sum(1 for line in open(opts.input,'r'))

    Q = lexsub.updater(num_iteration = 10)
    print("DONE!")

    write_wvec_to_file(wordVectors=Q, outputFile='data/glove.6B.100d.retrofit.txt')
    print("write new vectors in a file")

    os.system("python3 -m pymagnitude.converter -i data/glove.6B.100d.retrofit.txt -o data/glove.6B.100d.retrofit.magnitude")
