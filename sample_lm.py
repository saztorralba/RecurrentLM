import sys
import os
import math
import random
import numpy as np
import argparse

from utils.rnnlm_func import sample_model 

import torch

import warnings
warnings.filterwarnings('ignore')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test the simple recurrent LM by generating sentences sampled from the LM')
    parser.add_argument('--model_file', metavar='FILE', default=None, help='Full path to trained serialised model')
    parser.add_argument('--num_sequences',type=int,default=10,help='Number of sentences to generate')
    parser.add_argument('--verbose',type=int,default=0,help='Verbosity level (0: global results, 1: sentence results, 2: word results)')
    parser.add_argument('--seed',type=int,default=0,help='Seed to initialise the pseudo-random generators')
    parser.add_argument('--topk',type=int,default=100,help='Number words with the highest probability from which to draw the next word')
    parser.add_argument('--distribution',type=str,default='uniform',help='Type of distribution to draw the next word from (exponential or uniform)')
    parser.add_argument('--end_prob',type=float,default=0.1,help='Initial probability to end a sentence')
    parser.add_argument('--end_prob_rate',type=float,default=1.05,help='Increase of the probability to end a sentence for each new word')
    parser.add_argument('--start_with',type=str,default='',help='Initial sequence of words for the generated sequences')
    args = parser.parse_args()
    return args

def sample_lm(args):
    #Perform initialisations
    args.debug = min(max(args.verbose,0),2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #Read model
    model = torch.load(args.model_file)
    args.vocab = model.vocab
    args.vocab_list = {args.vocab[v]:v for v in args.vocab}
    args.start_token, args.end_token, args.unk_token = (model.start_token, model.end_token, model.unk_token)
    args.characters = model.characters
    args.start_with = (list(args.start_with.strip()) if args.characters else args.start_with.strip().split())
    args.start_with = [x if x in args.vocab else args.unk_token for x in args.start_with]

    #Sample requires sentences
    logprob,total = sample_model(model,args)
    ppl = math.pow(10.0,-logprob/total)
    print('file {0:s}: {1:d} sentences, {2:d} words'.format(args.model_file,args.num_sequences,total))
    print('logprob = {0:.2f}, ppl = {1:.2f}'.format(logprob,ppl))

if __name__ == '__main__':
    args = parse_arguments()
    sample_lm(args)

