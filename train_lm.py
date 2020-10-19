import sys
import os
import random
from tqdm import tqdm
import argparse

from utils.rnnlm_func import load_data, build_model, train_model, validate_model
from utils.lm_func import read_vocabulary, count_sequences

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore')

def make_folder_for_file(fileName):
    folder = os.path.dirname(fileName)
    if folder != '' and not os.path.isdir(folder):
        os.makedirs(folder)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a simple auto-regressive recurrent LM')
    parser.add_argument('--input_file', metavar='FILE', default=None, help='Full path to a file containing normalised sentences')
    parser.add_argument('--output_file', metavar='FILE', default=None, help='Full path to the output model file as a torch serialised object')
    parser.add_argument('--vocabulary',metavar='FILE',default=None,help='Full path to a file containing the vocabulary words')
    parser.add_argument('--cv_percentage',default=0.1,type=float,help='Amount of data to use for cross-validation')
    parser.add_argument('--epochs',type=int,default=10,help='Number of epochs to train')
    parser.add_argument('--batch_size',type=int,default=32,help='Batch size')
    parser.add_argument('--val_batch_size',type=int,default=100,help='Batch size for validation')
    parser.add_argument('--embedding_size',type=int,default=128,help='Size of the embedding layer')
    parser.add_argument('--hidden_size',type=int,default=128,help='Size of the hidden recurrent layers')
    parser.add_argument('--num_layers',type=int,default=1,help='Number of recurrent layers')
    parser.add_argument('--learning_rate',type=float,default=0.1,help='Learning rate')
    parser.add_argument('--seed',type=float,default=0,help='Random seed')
    parser.add_argument('--bptt',type=int,default=sys.maxsize,help='Length of segments for Truncated Back-Propagation Through Time')
    parser.add_argument('--max_length',type=int,default=sys.maxsize,help='Maximum length of sequences to use (longer sequences are discarded)')
    parser.add_argument('--ltype',type=str,default='lstm',help='Type of hidden layers to use ("rnn", "gru", "lstm")')
    parser.add_argument('--nonlinearity',type=str,default='relu',help='Non-linear function used in the recurrent layersi (for RNN)')
    parser.add_argument('--start_token',type=str,default='<s>',help='Word token used at the beginning of a sentence')
    parser.add_argument('--end_token',type=str,default='<\s>',help='Word token used at the end of a sentence')
    parser.add_argument('--unk_token',type=str,default='<UNK>',help='Word token used for out-of-vocabulary words')
    args = parser.parse_args()
    return args

def train_lm(args):
    #Initialisations
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    args.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    #Read vocabulary, sentences and load data
    args.vocab,args.characters = read_vocabulary(args)
    args.num_seq,args.max_words = count_sequences(args)
    trainset,validset = load_data(args, True)
    print('Number of training sequences: {0:d}'.format(trainset.shape[1]))
    print('Number of cross-validaton sequences: {0:d}'.format(validset.shape[1]))

    #Create model, optimiser and criterion
    model = build_model(args).to(args.device) 
    optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate)
    criterion = nn.NLLLoss(reduction='none').to(args.device)
    print('\nModel:')
    print(model)
    print('\n')

    #Train epochs
    for ep in range(1,args.epochs+1):
        print('Epoch {0:d} of {1:d}'.format(ep,args.epochs))
        train_model(trainset,model,optimizer,criterion,args)
        ppl = validate_model(validset,model,args)

        #Save intermediate models
        nfolder = os.path.dirname(args.output_file)
        nfile = nfolder+'/intermediate/model_epoch{0:02d}_ppl{1:0.2f}.pytorch'.format(ep,ppl)
        make_folder_for_file(nfile)
        torch.save(model,nfile)

    #Save final models
    model.cpu_hidden()
    model = model.cpu().eval()
    make_folder_for_file(args.output_file)
    torch.save(model,args.output_file)

if __name__ == '__main__':
    args = parse_arguments()
    train_lm(args)

