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

def load_data(fileName,num_seq,max_words,cv,vocab,num_words,stoken,etoken,utoken,max_len):
    dataset = num_words*torch.ones((max_words,num_seq),dtype=torch.long)
    datamask = torch.zeros((max_words,num_seq),dtype=torch.short)
    datalength = torch.zeros(num_seq,dtype=torch.long)
    idx = 0
    utoken_value = vocab[utoken]
    lines = [line for line in open(fileName)]
    for line in tqdm(lines,desc='Allocating data memory'):
        words = line.strip().split()
        if len(words)>0:
            if words[0] != stoken:
                words.insert(0,stoken)
            if words[-1] != etoken:
                words.append(etoken)
            if len(words) <= max_len:
                for jdx,word in enumerate(words):
                    dataset[jdx,idx] = vocab.get(word,utoken_value)
                datamask[0:len(words)-1,idx] = 1
                datalength[idx] = len(words)
                idx += 1

    idx = [i for i in range(num_seq)]
    random.shuffle(idx)
    trainidx = idx[0:int(num_seq*(1-cv))]
    valididx = idx[int(num_seq*(1-cv)):]

    trainset = dataset[:,trainidx]
    validset = dataset[:,valididx]
    trainmask = datamask[:,trainidx]
    validmask = datamask[:,valididx]
    trainlength = datalength[trainidx]
    validlength = datalength[valididx]
    return trainset, validset, trainmask, validmask, trainlength, validlength

def build_model(args,num_words,device):
    if args.ltype.lower() == 'lstm':
        model = LSTMLM(num_words, args.embedding_size, args.hidden_size, args.num_layers)
    elif args.ltype.lower() == 'rnn':
        model = RNNLM(num_words, args.embedding_size, args.hidden_size, args.num_layers, args.nonlinearity)
    elif args.ltype.lower() == 'gru':
        model = GRULM(num_words, args.embedding_size, args.hidden_size, args.num_layers)
    else:
        print('ERROR: Unknown layer type ({0:s})'.format(args.ltype))
        sys.exit()
    model = model.to(device)
    model.init_weights()
    return model

def load_test_data(fileName,num_seq,max_words,vocab,num_words,stoken,etoken,utoken):
    testset = num_words*torch.ones((max_words,num_seq),dtype=torch.long)
    testmask = torch.zeros((max_words,num_seq),dtype=torch.short)
    testlength = torch.zeros((num_seq),dtype=torch.long)
    idx=0
    utoken_value = vocab[utoken]
    lines = [line for line in open(fileName)]
    for line in tqdm(lines,desc='Allocating data memory'):
        words = line.strip().split()
        if len(words) > 0:
            if words[0] != stoken:
                words.insert(0,stoken)
            if words[-1] != etoken:
                words.append(etoken)
            for jdx,word in enumerate(words):
                testset[jdx,idx] = vocab.get(word,utoken_value)
            testmask[0:len(words)-1,idx] = 1
            testlength[idx] = len(words)
            idx += 1
    return testset, testmask, testlength

def train_model(trainset,trainmask,trainlength,model,optimizer,criterion,device,args):
    trainlen = trainset.shape[1]
    nbatches = math.ceil(trainlen/args.batch_size)
    total_loss = 0
    total_backs = 0
    with tqdm(total=nbatches) as pbar:
        model = model.train()
        for b in range(nbatches):
            X = trainset[:,b*args.batch_size:min(trainlen,(b+1)*args.batch_size)].clone().long()
            mask = trainmask[:,b*args.batch_size:min(trainlen,(b+1)*args.batch_size)].clone().long()
            seq_length = trainlength[b*args.batch_size:min(trainlen,(b+1)*args.batch_size)].clone().long()
            #get the batch
            X,Y,mask,ordered_seq_length,_ = process_batch(X,mask,seq_length)
            X = X.to(device)
            Y = Y.to(device)
            mask = mask.to(device)
            #propagate forward
            model.init_hidden(X.size(1))
            while ordered_seq_length[0]>0:
                tmp_X,tmp_Y,tmp_ordered_seq_length,tmp_mask,X,Y,ordered_seq_length,mask,valid = do_bptt(X,Y,ordered_seq_length,mask,args.bptt)
                model.cut_hidden(valid)
                posteriors = model(tmp_X,tmp_ordered_seq_length)
                #flatten outputs and targets
                flat_posteriors = posteriors.view(posteriors.size(0)*posteriors.size(1),posteriors.size(2))
                flat_mask = tmp_mask.view(tmp_mask.size(0)*tmp_mask.size(1))
                flat_Y = tmp_Y.view(tmp_Y.size(0)*tmp_Y.size(1))
                flat_Y = flat_Y*flat_mask
                #compute non reduced loss
                flat_loss = criterion(flat_posteriors,flat_Y)
                flat_loss = flat_loss*flat_mask.float()
                #get the averaged loss
                mean_loss = torch.sum(flat_loss)/torch.sum(flat_mask)
                #backpropagate
                optimizer.zero_grad()
                mean_loss.backward()
                optimizer.step()
                if total_backs == 100:
                    total_loss = total_loss*0.99+mean_loss.detach().cpu().numpy()
                else:
                    total_loss += mean_loss.detach().cpu().numpy()
                    total_backs += 1
                model.detach_hidden()
            pbar.set_description(f'Training epoch. Loss {total_loss/(total_backs+1):.2f}')
            pbar.update()

def validate_model(validset,validmask,validlength,model,device,num_words,args):
    validlen = validset.shape[1]
    logprob = 0
    total = 0
    nbatches = math.ceil(validlen/args.val_batch_size)
    with torch.no_grad():
        with tqdm(total=nbatches) as pbar:
            model = model.eval()
            for b in range(nbatches):
                X = validset[:,b*args.val_batch_size:min(validlen,(b+1)*args.val_batch_size)].clone().long()
                mask = validmask[:,b*args.val_batch_size:min(validlen,(b+1)*args.val_batch_size)].clone().long()
                seq_length = validlength[b*args.val_batch_size:min(validlen,(b+1)*args.val_batch_size)].clone().long()
                #get the batch
                X,Y,mask,ordered_seq_length,_ = process_batch(X,mask,seq_length)
                X = X.to(device)
                #propagate forward
                model.init_hidden(X.size(1))
                posteriors = model(X,ordered_seq_length)
                #flatten outputs and targets
                posteriors = posteriors.cpu()
                flat_posteriors = posteriors.view(posteriors.size(0)*posteriors.size(1),posteriors.size(2))
                flat_mask = mask.view(mask.size(0)*mask.size(1))#.expand(-1,posteriors.size(2))
                flat_Y = torch.clamp(Y.view(Y.size(0)*Y.size(1)),max=num_words-1)
                #accumulate ppl per event
                logprob = logprob-torch.sum(flat_posteriors[torch.arange(flat_posteriors.size(0)).long(),flat_Y]*flat_mask.float())
                total = total+torch.sum(mask)
                ppl = math.pow(10.0,(logprob.numpy()/math.log(10.0))/(total.numpy()))
                pbar.set_description(f'Evaluating epoch. Perplexity {ppl:.2f}')
                pbar.update()
    return ppl

def test_model(testset,testmask,testlength,model,orig_sent,num_words,args):
    num_seq = testset.shape[1]
    logprob = 0
    total = 0
    nbatches = math.ceil(num_seq/args.batch_size)
    sidx = 0
    ppls = list()
    with torch.no_grad():
        model = model.eval()
        for b in range(nbatches):
            X = testset[:,b*args.batch_size:min(num_seq,(b+1)*args.batch_size)].clone().long()
            mask = testmask[:,b*args.batch_size:min(num_seq,(b+1)*args.batch_size)].clone().long()
            seq_length = testlength[b*args.batch_size:min(num_seq,(b+1)*args.batch_size)].clone().long()
            #get the batch
            X,Y,mask,ordered_seq_length, dec_index = process_batch(X,mask,seq_length)
            #propagate forward
            model.init_hidden(X.size(1))
            posteriors = model(X,ordered_seq_length)
            #flatten outputs and targets
            flat_posteriors = posteriors.view(posteriors.size(0)*posteriors.size(1),posteriors.size(2))
            flat_mask = mask.view(mask.size(0)*mask.size(1))#.expand(-1,posteriors.size(2))
            flat_Y = torch.clamp(Y.view(Y.size(0)*Y.size(1)),max=num_words-1)
            ologprob = flat_posteriors[torch.arange(flat_posteriors.size(0)).long(),flat_Y]*flat_mask.float()
            ologprob = ologprob.view(posteriors.size(0),posteriors.size(1))
            slogprob = -torch.sum(ologprob,0)
            smask = torch.sum(mask,0)
            for s in range(seq_length.shape[0]):
                ridx = torch.nonzero(dec_index==s)[0,0].numpy()
                if args.debug > 0:
                    print(orig_sent[sidx])
                    if args.debug == 2:
                        words = sent[sidx].split()
                        for w in range(1,len(words)):
                            if w == 1:
                                print('\tp( {0:s} | {1:s} )\t= {2:f} [ {3:f} ]'.format(words[w],words[w-1],math.pow(10.0,(ologprob[w-1,ridx].numpy()/math.log(10.0))),ologprob[w-1,ridx].numpy()/math.log(10.0)))
                            else:
                                print('\tp( {0:s} | {1:s} ...)\t= {2:f} [ {3:f} ]'.format(words[w],words[w-1],math.pow(10.0,(ologprob[w-1,ridx].numpy()/math.log(10.0))),ologprob[w-1,ridx].numpy()/math.log(10.0)))
                    print("1 sentences, {0:d} words".format(len(orig_sent[sidx].split())))
                    print('logprob = {0:.2f}, ppl = {1:.2f}'.format(-slogprob[ridx].numpy()/math.log(10.0),math.pow(10.0,(slogprob[ridx].numpy()/math.log(10.0))/smask[ridx].numpy())))
                ppls.append(math.pow(10.0,(slogprob[ridx].numpy()/math.log(10.0))/smask[ridx].numpy()))
                sidx += 1
            #accumulate ppl per event
            logprob = logprob-torch.sum(flat_posteriors[torch.arange(flat_posteriors.size(0)).long(),flat_Y]*flat_mask.float())
            total = total+torch.sum(mask)
    return logprob, total

def sample_model(model,vocab,vocab_list,num_words,args):
    with torch.no_grad():
        model = model.eval()
        logprob = 0
        total = 0
        for b in range(args.num_sequences):
            X = vocab[args.start_token]*torch.ones((1,1),dtype=torch.long)
            seq_length = torch.ones((1),dtype=torch.long)
            orig_words = []
            words = [args.start_token]
            probs = []
            model.init_hidden(1)
            end_prob = args.end_prob
            start_with = args.start_with
            while words[-1] != args.end_token:
                posteriors = model(X,seq_length)
                posteriors = posteriors[0,0,:]
                posteriors = torch.pow(10.0,posteriors/math.log(10.0))
                if posteriors[vocab[args.end_token]] >= end_prob:
                    words.append(args.end_token)
                    probs.append(posteriors[vocab[args.end_token]].numpy())
                    break
                if start_with is not None and start_with!="":
                    next_idx = np.array(vocab[start_with.split()[0]])
                    start_with = ' '.join(start_with.split()[1:])
                else:
                    posteriors[vocab[args.unk_token]] = 0
                    dec_index = torch.argsort(posteriors,descending=True)
                    if args.topk <= posteriors.shape[0]:
                        dec_index = dec_index[0:args.topk]
                    sposteriors = posteriors[dec_index].numpy()
                    if args.distribution == 'exponential':
                        val = 2
                        while val > 1:
                            val = np.random.exponential(1/3)
                    elif args.distribution == 'uniform':
                        val=random.random()
                    else:
                        val=random.random()
                    next_idx = dec_index[np.where((np.cumsum(sposteriors)/np.sum(sposteriors))>=val)[0][0]].numpy()
                    if vocab_list[int(next_idx)] == args.end_token:
                        words.append(args.end_token)
                        probs.append(posteriors[vocab[args.end_token]].numpy())
                        break
                X = np.asscalar(next_idx)*torch.ones((1,1),dtype=torch.long)
                words.append(vocab_list[int(next_idx)])
                orig_words.append(vocab_list[int(next_idx)])
                probs.append(posteriors[next_idx].numpy())
                end_prob /= args.end_prob_rate
            probs=np.array(probs)
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
            #accumulate ppl per event
            logprob = logprob+np.sum(np.log10(probs))
            total = total+len(orig_words)
    return logprob, total

def process_batch(X,mask,seq_length):
    #order the sequences by decreasing length
    ordered_seq_length, dec_index = seq_length.sort(descending=True)
    X = X[:,dec_index]
    mask = mask[:,dec_index]
    #remove excessive padding
    max_seq_length = torch.max(seq_length)
    X = X[0:max_seq_length]
    mask = mask[0:max_seq_length]
    #create targets by moving inputs one step backward
    Y = X[1:X.size(0)].clone()
    X = X[0:X.size(0)-1]
    mask = mask[0:X.size(0)]
    ordered_seq_length = ordered_seq_length-1
    return X, Y, mask, ordered_seq_length, dec_index

def do_bptt(X,Y,seq_length,mask,bptt):
    new_seq_length = torch.clamp(seq_length,max=bptt)
    seq_length = torch.clamp(seq_length-bptt,min=0)
    valid = torch.max(torch.nonzero(new_seq_length))+1
    new_seq_length = new_seq_length[0:valid]
    seq_length = seq_length[0:valid]
    limit = min(bptt,new_seq_length[0])
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

    return new_X, new_Y, new_seq_length, new_mask.contiguous(), X, Y, seq_length, mask.contiguous(), valid
