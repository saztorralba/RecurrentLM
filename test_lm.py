import sys
import os
import math
import numpy as np
from tqdm import tqdm
import argparse

from utils.rnnlm_func import load_data, test_model
from utils.lm_func import read_sentences
import torch

import warnings
warnings.filterwarnings('ignore')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test the simple auto-regressive recurrent LM calculating perplexity for test sentences ')
    parser.add_argument('--input_file', metavar='FILE', default=None, help='Full path to a file containing normalised sentences')
    parser.add_argument('--model_file', metavar='FILE', default=None, help='Full path to trained serialised model')
    parser.add_argument('--batch_size',type=int,default=64,help='Batch size')
    parser.add_argument('--verbose',default=0,type=int,choices=[0,1,2],help='Verbosity level (0, 1 or 2)')
    args = parser.parse_args()
    args = vars(args)
    return args

def test_lm(args):
    #Read model and initializations
    args['max_length'] = sys.maxsize
    model = torch.load(args['model_file'])
    args['vocab'] = model.vocab
    args['start_token'], args['end_token'], args['unk_token'] = (model.start_token, model.end_token, model.unk_token)
    args['characters'] = model.characters

    #Read sentences to test and load data
    orig_sent,sent = read_sentences(**args)
    args['num_seq'] = len(sent)
    args['max_words'] = max([len(s) for s in sent])
    testset = load_data(False, **args)

    #Compute perplexities
    logprob,total = test_model(testset,model,orig_sent,sent,**args)
    ppl = math.pow(10.0,(logprob.numpy()/math.log(10.0))/(total.numpy()))
    print('file {0:s}: {1:d} sentences, {2:d} words'.format(args['input_file'],args['num_seq'],total-args['num_seq']))
    print('logprob = {0:.2f}, ppl = {1:.2f}'.format(-logprob.numpy()/math.log(10.0),ppl))

if __name__ == '__main__':
    args=parse_arguments()
    test_lm(args)

