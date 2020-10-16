import sys
import os
import math
import numpy as np
from tqdm import tqdm
import argparse

from utils.rnnlm_func import process_batch, load_test_data, test_model
from utils.lm_func import read_vocabulary, count_sequences, read_sentences
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test the simple recurrent LM calculating perplexity for test sentences ')
    parser.add_argument('--input_file', metavar='FILE', default=None, help='Full path to a file containing normalised sentences')
    parser.add_argument('--model_file', metavar='FILE', default=None, help='Full path to trained serialised model')
    parser.add_argument('--vocabulary',metavar='FILE',default=None,help='Full path to a file containing the vocabulary words')
    parser.add_argument('--batch_size',type=int,default=64,help='Batch size')
    parser.add_argument('--start_token',type=str,default='<s>',help='Word token used at the beginning of a sentence')
    parser.add_argument('--end_token',type=str,default='<\s>',help='Word token used at the end of a sentence')
    parser.add_argument('--unk_token',type=str,default='<UNK>',help='Word token used for out-of-vocabulary words')
    parser.add_argument('--verbose',type=int,default=0,help='Verbosity level (0: global results, 1: sentence results, 2: word results)')
    args = parser.parse_args()
    return args

def test_lm(args):
    args.debug = min(max(args.verbose,0),2)

    vocab,num_words = read_vocabulary(args.vocabulary,args.start_token,args.end_token,args.unk_token)

    num_seq,max_words = count_sequences(args.input_file,args.start_token,args.end_token,sys.maxsize)
    orig_sent,sent = read_sentences(args.input_file,vocab,args.start_token,args.end_token,args.unk_token)

    testset,testmask,testlength = load_test_data(args.input_file,num_seq,max_words,vocab,num_words,args.start_token,args.end_token,args.unk_token)

    model = torch.load(args.model_file)
    model = model.cpu()

    logprob,total = test_model(testset,testmask,testlength,model,orig_sent,num_words,args)

    ppl = math.pow(10.0,(logprob.numpy()/math.log(10.0))/(total.numpy()))
    print('file {0:s}: {1:d} sentences, {2:d} words'.format(args.input_file,num_seq,total-num_seq))
    print('logprob = {0:.2f}, ppl = {1:.2f}'.format(-logprob.numpy()/math.log(10.0),ppl))

if __name__ == '__main__':
    args=parse_arguments()
    test_lm(args)

