import sys
import logging
import os.path
from gensim.models import word2vec
import cPickle

##### Launch in terminal with python accuracy_w2v.py questions.txt "the_model_you_want_to_test"

if __name__ == '__main__':
	program = os.path.basename(sys.argv[0])
	logger = logging.getLogger(program)
	logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
	logging.root.setLevel(level=logging.INFO)
	logger.info("running %s" % ' '.join(sys.argv))

	# check and process input arguments
	
	questions, inp = sys.argv[1:3]

	evals = open(questions, 'r').readlines()
	num_sections = len([l for l in evals if l.startswith(':')])
	print('total evaluation sentences: {} '.format(len(evals) - num_sections))

	### Load model ###
	model = word2vec.Word2Vec.load(inp)
	
	sub_score = {}
	def w2v_model_accuracy(model):
		accuracy = model.accuracy(questions)
		for i in range(len(accuracy)) :
			sumc_int = float(len(accuracy[i]['correct']))
			sumi_int = float(len(accuracy[i]['incorrect']))
			sumt_int = sumc_int + sumi_int
			score = sumc_int / sumt_int *100.0
			sub_score[accuracy[i]['section']] = score
	  
	    	print sub_score
		
		with open('results'+str(inp), 'wb') as handle:
				cPickle.dump(sub_score, handle)
		return accuracy

	### compute score ###
 	results = w2v_model_accuracy(model)
