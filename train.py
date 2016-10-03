import logging
import os.path
import sys
import multiprocessing
 
from gensim.corpora import  WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
 
##### Launch in terminal with python train_w2v_wiki.py "your text corpus" "the output model name"

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments
 
    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    inp, outp = sys.argv[1:3]
 
    model = Word2Vec(LineSentence(inp), sg = 1, hs = 0, cbow_mean = 1, alpha = 0.05, window = 5, size=400, iter = 20, sample = 1e-4, negative = 15, min_count=5, workers=multiprocessing.cpu_count())
 
    model.save(outp)