import numpy as np
import torch
import torch.nn as nn
import sys

class GRULM(nn.Module):
    def __init__(self,vocab,embed_dim=512,hid_dim=512,n_layers=1,tokens=[],characters=False):
        
        super(GRULM, self).__init__()
        #Base variables
        self.vocab = vocab
        self.in_dim = len(self.vocab)
        self.start_token = tokens[0]
        self.end_token = tokens[1]
        self.unk_token = tokens[2]
        self.characters = characters
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        #Define the embedding layer
        self.embed = nn.Embedding(self.in_dim+1,self.embed_dim,padding_idx=self.in_dim)
        #Define the GRU layer
        self.gru= nn.GRU(input_size=self.embed_dim,hidden_size=self.hid_dim,num_layers=n_layers)
        #Define the output layer
        self.linear = nn.Linear(self.hid_dim,self.in_dim)
        #Define the softmax layer
        self.softmax = nn.LogSoftmax()
        
    def init_weights(self):
        #Randomly initialise all parameters
        torch.nn.init.xavier_uniform_(self.embed.weight)
        for i in range(self.n_layers):
            torch.nn.init.xavier_uniform_(getattr(self.gru,'weight_hh_l'+str(i)))
            torch.nn.init.xavier_uniform_(getattr(self.gru,'weight_ih_l'+str(i)))
            torch.nn.init.uniform_(getattr(self.gru,'bias_hh_l'+str(i)))
            torch.nn.init.uniform_(getattr(self.gru,'bias_ih_l'+str(i)))
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.uniform_(self.linear.bias)
    
    def forward(self, inputs, lengths):
        #Inputs are size (LxBx1)
        #Forward embedding layer
        emb = self.embed(inputs)
        #Embeddings are size (LxBxself.embed_dim)

        #Pack the sequences for GRU
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths)
        #Forward the GRU
        packed_rec, self.hidden = self.gru(packed,self.hidden)
        #Unpack the sequences
        rec, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_rec)
        #Hidden outputs are size (LxBxself.hidden_size)
        
        #Flatten for the output layer
        flat_rec = rec.view(rec.size(0)*rec.size(1), rec.size(2))
        #Forward the output layer and the softmax
        flat_out = self.softmax(self.linear(flat_rec))
        #Unflatten the output
        out = flat_out.view(inputs.size(0),inputs.size(1),flat_out.size(1))
        #Outputs are size (LxBxself.in_dim)
        
        return out
    
    def init_hidden(self, bsz):
        #Initialise the hidden state
        weight = next(self.parameters())
        self.hidden = weight.new_zeros(self.n_layers, bsz, self.hid_dim)

    def detach_hidden(self):
        #Detach the hidden state
        self.hidden=self.hidden.detach()

    def cpu_hidden(self):
        #Set the hidden state to CPU
        self.hidden=self.hidden.detach().cpu()

    def cut_hidden(self,valid):
        #Reduce the batch size of the hidden state
        self.hidden=self.hidden[:,0:valid,:].contiguous()

