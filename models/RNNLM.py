import numpy as np
import torch
import torch.nn as nn
import sys

class RNNLM(nn.Module):
    def __init__(self,in_dim,embed_dim=512,hid_dim=512,n_layers=1,nonlinearity='relu'):
        
        super(RNNLM, self).__init__()
        #base variables
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.nonlinearity=nonlinearity
        
        #define the embedding layer
        self.embed = nn.Embedding(self.in_dim+1,self.embed_dim,padding_idx=self.in_dim)
        #define the lstm layer
        self.rnn= nn.RNN(input_size=self.embed_dim,hidden_size=self.hid_dim,num_layers=n_layers,nonlinearity=self.nonlinearity.lower())
        #define the output layer
        self.linear = nn.Linear(self.hid_dim,self.in_dim)
        #define the softmax layer
        self.softmax = nn.LogSoftmax()
        
    def init_weights(self):
        #randomly initialise all parameters
        torch.nn.init.xavier_uniform_(self.embed.weight)
        for i in range(self.n_layers):
            torch.nn.init.xavier_uniform_(getattr(self.rnn,'weight_hh_l'+str(i)))
            torch.nn.init.xavier_uniform_(getattr(self.rnn,'weight_ih_l'+str(i)))
            torch.nn.init.uniform_(getattr(self.rnn,'bias_hh_l'+str(i)))
            torch.nn.init.uniform_(getattr(self.rnn,'bias_ih_l'+str(i)))
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.uniform_(self.linear.bias)
    
    def forward(self, inputs, lengths):
        #inputs is size (LxBx1)
        #forward embedding layer
        emb = self.embed(inputs)
        #emb is (LxBxself.embed_dim)

        #pack the sequences for LSTM
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths)
        #forward the LSTM
        packed_rec, self.hidden = self.rnn(packed,self.hidden)
        #unpack the sequences
        rec, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_rec)
        #rec is (LxBxself.hidden_size)
        
        #flatten for the output layer
        flat_rec = rec.view(rec.size(0)*rec.size(1), rec.size(2))
        #forward the output layer and the softmax
        flat_out = self.softmax(self.linear(flat_rec))
        #unflatten the output
        out = flat_out.view(inputs.size(0),inputs.size(1),flat_out.size(1))
        #out is (LxBxself.in_dim)
        
        return out
    
    def init_hidden(self, bsz):
        #initialise the LSTM state
        weight = next(self.parameters())
        self.hidden = weight.new_zeros(self.n_layers, bsz, self.hid_dim)

    def detach_hidden(self):
        self.hidden=self.hidden.detach()

    def cpu_hidden(self):
        self.hidden=self.hidden.detach().cpu()

    def cut_hidden(self,valid):
        self.hidden=self.hidden[:,0:valid,:].contiguous()

