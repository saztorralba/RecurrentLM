import sys
import random
from tqdm import tqdm
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.LSTMLM import LSTMLM
from models.RNNLM import RNNLM
from models.GRULM import GRULM

#Load data in memory from an input text (when cv is True, perform cv split)
def load_data(args, cv=False):
    dataset = len(args.vocab)*torch.ones((args.max_words,args.num_seq),dtype=torch.long)
    idx = 0
    utoken_value = args.vocab[args.unk_token]
    lines = [line for line in open(args.input_file)]
    for line in tqdm(lines,desc='Allocating data memory'):
        words = (list(line.strip()) if args.characters else line.strip().split())
        if len(words)>0:
            if words[0] != args.start_token:
                words.insert(0,args.start_token)
            if words[-1] != args.end_token:
                words.append(args.end_token)
            if len(words) <= args.max_length:
                for jdx,word in enumerate(words):
                    dataset[jdx,idx] = args.vocab.get(word,utoken_value)
                idx += 1

    if cv == False:
        return dataset

    idx = [i for i in range(args.num_seq)]
    random.shuffle(idx)
    trainset = dataset[:,idx[0:int(args.num_seq*(1-args.cv_percentage))]]
    validset = dataset[:,idx[int(args.num_seq*(1-args.cv_percentage)):]]
    return trainset, validset

#Create the model architecture based on input parameters
def build_model(args):
    if args.ltype.lower() == 'lstm':
        model = LSTMLM(args.vocab, args.embedding_size, args.hidden_size, args.num_layers, [args.start_token,args.end_token,args.unk_token],args.characters)
    elif args.ltype.lower() == 'rnn':
        model = RNNLM(args.vocab, args.embedding_size, args.hidden_size, args.num_layers, args.nonlinearity, [args.start_token,args.end_token,args.unk_token],args.characters)
    elif args.ltype.lower() == 'gru':
        model = GRULM(args.vocab, args.embedding_size, args.hidden_size, args.num_layers, [args.start_token,args.end_token,args.unk_token],args.characters)
    else:
        print('ERROR: Unknown layer type ({0:s})'.format(args.ltype))
        sys.exit()
    model.init_weights()
    return model

#Perform backpropagation training on all batches
def train_model(trainset,model,optimizer,criterion,args):
    trainlen = trainset.shape[1]
    nbatches = math.ceil(trainlen/args.batch_size)
    total_loss = 0
    total_backs = 0
    with tqdm(total=nbatches) as pbar:
        model = model.train()
        for b in range(nbatches):
            #Data batch
            X = trainset[:,b*args.batch_size:min(trainlen,(b+1)*args.batch_size)].clone().long().to(args.device)
            mask = torch.clamp(len(args.vocab)-X,max=1)
            seq_length = torch.sum(mask,dim=0)
            #Reorder the batch by sequence length
            X,Y,mask,ordered_seq_length,_ = process_batch(X,mask,seq_length)
            model.init_hidden(X.size(1))
            while ordered_seq_length[0]>0:
                #Loop, doing multiple passes with truncated sequences (if truncated bptt is used)
                tmp_X,tmp_Y,tmp_ordered_seq_length,tmp_mask,X,Y,ordered_seq_length,mask,valid = do_bptt(X,Y,ordered_seq_length,mask,args.bptt)
                model.cut_hidden(valid)
                #Forward model, get posteriors
                posteriors = model(tmp_X,tmp_ordered_seq_length)
                #Flatten outputs and targets
                flat_posteriors = posteriors.view(posteriors.size(0)*posteriors.size(1),posteriors.size(2))
                flat_mask = tmp_mask.view(tmp_mask.size(0)*tmp_mask.size(1))
                flat_Y = tmp_Y.view(tmp_Y.size(0)*tmp_Y.size(1))
                flat_Y = flat_Y*flat_mask
                #Compute non reduced loss
                flat_loss = criterion(flat_posteriors,flat_Y)
                flat_loss = flat_loss*flat_mask.float()
                #Get the averaged/reduced loss using the mask
                mean_loss = torch.sum(flat_loss)/torch.sum(flat_mask)
                #Backpropagate
                optimizer.zero_grad()
                mean_loss.backward()
                optimizer.step()
                #Estimate the latest loss
                if total_backs == 100:
                    total_loss = total_loss*0.99+mean_loss.detach().cpu().numpy()
                else:
                    total_loss += mean_loss.detach().cpu().numpy()
                    total_backs += 1
                model.detach_hidden()
            pbar.set_description(f'Training epoch. Loss {total_loss/(total_backs+1):.2f}')
            pbar.update()

#Do forward passing and perplexity estimation in all batches
def validate_model(validset,model,args):
    validlen = validset.shape[1]
    logprob = 0
    total = 0
    nbatches = math.ceil(validlen/args.val_batch_size)
    with torch.no_grad():
        with tqdm(total=nbatches) as pbar:
            model = model.eval()
            for b in range(nbatches):
                #Data batch
                X = validset[:,b*args.val_batch_size:min(validlen,(b+1)*args.val_batch_size)].clone().long().to(args.device)
                mask = torch.clamp(len(args.vocab)-X,max=1)
                seq_length = torch.sum(mask,dim=0)
                #Reorder the batch by sequence length
                X,Y,mask,ordered_seq_length,_ = process_batch(X,mask,seq_length)
                #Propagate forward
                model.init_hidden(X.size(1))
                posteriors = model(X,ordered_seq_length)
                #Flatten outputs and targets
                flat_posteriors = posteriors.view(posteriors.size(0)*posteriors.size(1),posteriors.size(2))
                flat_mask = mask.view(mask.size(0)*mask.size(1))
                flat_Y = torch.clamp(Y.view(Y.size(0)*Y.size(1)),max=len(args.vocab)-1)
                #Accumulate ppl per event
                logprob = logprob-torch.sum(flat_posteriors[torch.arange(flat_posteriors.size(0)).long(),flat_Y]*flat_mask.float())
                total = total+torch.sum(mask)
                ppl = math.pow(10.0,(logprob.cpu().numpy()/math.log(10.0))/(total.cpu().numpy()))
                pbar.set_description(f'Evaluating epoch. Perplexity {ppl:.2f}')
                pbar.update()
    return ppl

#Do forward pass and estimate perplexity for set of sentences
def test_model(testset,model,orig_sent,sent,args):
    num_seq = testset.shape[1]
    logprob = 0
    total = 0
    nbatches = math.ceil(args.num_seq/args.batch_size)
    sidx = 0
    ppls = list()
    with torch.no_grad():
        model = model.eval()
        for b in range(nbatches):
            #Data batch
            X = testset[:,b*args.batch_size:min(num_seq,(b+1)*args.batch_size)].clone().long()
            mask = torch.clamp(len(args.vocab)-X,max=1)
            seq_length = torch.sum(mask,dim=0)
            ##Reorder the batch by sequence length
            X,Y,mask,ordered_seq_length, dec_index = process_batch(X,mask,seq_length)
            #Propagate forward
            model.init_hidden(X.size(1))
            posteriors = model(X,ordered_seq_length)
            #Flatten outputs and targets
            flat_posteriors = posteriors.view(posteriors.size(0)*posteriors.size(1),posteriors.size(2))
            flat_mask = mask.view(mask.size(0)*mask.size(1))
            flat_Y = torch.clamp(Y.view(Y.size(0)*Y.size(1)),max=len(args.vocab)-1)
            ologprob = flat_posteriors[torch.arange(flat_posteriors.size(0)).long(),flat_Y]*flat_mask.float()
            ologprob = ologprob.view(posteriors.size(0),posteriors.size(1))
            slogprob = -torch.sum(ologprob,0)
            smask = torch.sum(mask,0)
            for s in range(seq_length.shape[0]):
                ridx = torch.nonzero(dec_index==s)[0,0].numpy()
                #If verbosity level is required, show results per sentence or word
                if args.debug > 0:
                    print(" ".join(orig_sent[sidx]))
                    if args.debug == 2:
                        words = sent[sidx]
                        for w in range(1,len(words)):
                            if w == 1:
                                print('\tp( {0:s} | {1:s} )\t= {2:f} [ {3:f} ]'.format(words[w],words[w-1],math.pow(10.0,(ologprob[w-1,ridx].numpy()/math.log(10.0))),ologprob[w-1,ridx].numpy()/math.log(10.0)))
                            else:
                                print('\tp( {0:s} | {1:s} ...)\t= {2:f} [ {3:f} ]'.format(words[w],words[w-1],math.pow(10.0,(ologprob[w-1,ridx].numpy()/math.log(10.0))),ologprob[w-1,ridx].numpy()/math.log(10.0)))
                    print("1 sentences, {0:d} words".format(len(orig_sent[sidx])))
                    print('logprob = {0:.2f}, ppl = {1:.2f}'.format(-slogprob[ridx].numpy()/math.log(10.0),math.pow(10.0,(slogprob[ridx].numpy()/math.log(10.0))/smask[ridx].numpy())))
                ppls.append(math.pow(10.0,(slogprob[ridx].numpy()/math.log(10.0))/smask[ridx].numpy()))
                sidx += 1
            #Accumulate global perplexity
            logprob = logprob-torch.sum(flat_posteriors[torch.arange(flat_posteriors.size(0)).long(),flat_Y]*flat_mask.float())
            total = total+torch.sum(mask)
    return logprob, total

#Generate a set of sentences
def sample_model(model,args):
    with torch.no_grad():
        model = model.eval()
        logprob = 0
        total = 0
        for b in range(args.num_sequences):
            #Initialise first symbol as start of sentence token
            X = args.vocab[args.start_token]*torch.ones((1,1),dtype=torch.long)
            seq_length = torch.ones((1),dtype=torch.long)
            orig_words = []
            words = [args.start_token]
            probs = []
            model.init_hidden(1)
            end_prob = args.end_prob
            start_with = args.start_with
            #Iterate until reaching end of sentence token
            while words[-1] != args.end_token:
                #Forward the last symbol
                posteriors = model(X,seq_length)
                posteriors = posteriors[0,0,:]
                #Get linear posteriors probabilities
                posteriors = torch.pow(10.0,posteriors/math.log(10.0))
                #Evaluate if the probability of the end of sentence is higher than current threshold
                if posteriors[args.vocab[args.end_token]] >= end_prob:
                    words.append(args.end_token)
                    probs.append(posteriors[args.vocab[args.end_token]].numpy())
                    break
                #If a prompt was given, advance to the next word in the prompt
                if start_with is not None and len(start_with)>0:
                    next_idx = np.array(args.vocab[start_with[0]])
                    start_with = start_with[1:]
                else:
                    #Remove the probability density of the unknown token
                    posteriors[args.vocab[args.unk_token]] = 0
                    #Sort words by posterior probability
                    dec_index = torch.argsort(posteriors,descending=True)
                    #Take only the top-K most probablr
                    if args.topk <= posteriors.shape[0]:
                        dec_index = dec_index[0:args.topk]
                    sposteriors = posteriors[dec_index].numpy()
                    #Get a random number from a distribution
                    if args.distribution == 'exponential':
                        val = 2
                        while val > 1:
                            val = np.random.exponential(1/3)
                    elif args.distribution == 'uniform':
                        val=random.random()
                    else:
                        val=random.random()
                    #Find the next word based on the random number and the probability densities of the words
                    next_idx = dec_index[np.where((np.cumsum(sposteriors)/np.sum(sposteriors))>=val)[0][0]].numpy()
                    if args.vocab_list[int(next_idx)] == args.end_token:
                        words.append(args.end_token)
                        probs.append(posteriors[args.vocab[args.end_token]].numpy())
                        break
                #Append the new word and update end of sentence probability
                X = np.asscalar(next_idx)*torch.ones((1,1),dtype=torch.long)
                words.append(args.vocab_list[int(next_idx)])
                orig_words.append(args.vocab_list[int(next_idx)])
                probs.append(posteriors[next_idx].numpy())
                end_prob /= args.end_prob_rate
            probs=np.array(probs)
            #Print sentence and perplexity if required
            if args.characters:
                print("".join(orig_words))
            else:
                print(" ".join(orig_words))
            if args.debug > 0:
                if args.debug == 2:
                    for w in range(1,len(words)):
                        if w == 1:
                            print('\tp( {0:s} | {1:s} )\t= {2:f} [ {3:f} ]'.format(words[w],words[w-1],probs[w-1],math.log10(probs[w-1])))
                        else:
                            print('\tp( {0:s} | {1:s} ...)\t= {2:f} [ {3:f} ]'.format(words[w],words[w-1],probs[w-1],math.log10(probs[w-1])))
                print("1 sentences, {0:d} words".format(len(orig_words)))
                print('logprob = {0:.2f}, ppl = {1:.2f}'.format(np.sum(np.log10(probs)),math.pow(10.0,-np.sum(np.log10(probs))/len(orig_words))))
            #Accumulate global perplexity
            logprob = logprob+np.sum(np.log10(probs))
            total = total+len(orig_words)
    return logprob, total

#Process a batch and reorder by sequence length
def process_batch(X,mask,seq_length):
    #Unmask the end of sentence element
    mask[seq_length-1,[i for i in range(mask.shape[1])]] = 0
    #Order the sequences by decreasing length
    ordered_seq_length, dec_index = seq_length.sort(descending=True)
    X = X[:,dec_index]
    mask = mask[:,dec_index]
    #Remove excessive padding
    max_seq_length = torch.max(seq_length)
    X = X[0:max_seq_length]
    mask = mask[0:max_seq_length]
    #Create targets by moving inputs one step backward
    Y = X[1:X.size(0)].clone()
    X = X[0:X.size(0)-1]
    mask = mask[0:X.size(0)]
    ordered_seq_length = ordered_seq_length-1
    return X, Y, mask, ordered_seq_length, dec_index

#Truncate the sequence for truncated backpropagation through time
def do_bptt(X,Y,seq_length,mask,bptt):
    #Clamp the sequence length to the bptt value
    new_seq_length = torch.clamp(seq_length,max=bptt)
    seq_length = torch.clamp(seq_length-bptt,min=0)
    #Remove sequences that have finished in previous truncations
    valid = torch.max(torch.nonzero(new_seq_length))+1
    new_seq_length = new_seq_length[0:valid]
    seq_length = seq_length[0:valid]
    limit = min(bptt,new_seq_length[0])
    #Chop input data and masks
    new_X = X[0:limit,0:valid].contiguous()
    if seq_length[0]>0:
        X = X[limit:,0:valid].contiguous()
    if Y.dim() == 2:
        new_Y = Y[0:limit,0:valid].contiguous()
        if seq_length[0]>0:
            Y = Y[limit:,0:valid].contiguous()
    new_mask = mask[0:limit,0:valid].contiguous()
    if seq_length[0]>0:
        mask = mask[limit:,0:valid]

    #Return the truncated sequence for backpropagation and the remainder for further loops
    return new_X, new_Y, new_seq_length, new_mask.contiguous(), X, Y, seq_length, mask.contiguous(), valid
