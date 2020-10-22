import numpy as np
import torch
import torch.nn as nn
import sys

class RNNLM(nn.Module):
    def __init__(self, **kwargs):
        
        super(RNNLM, self).__init__()
        #Base variables
        self.vocab = kwargs['vocab']
        self.in_dim = len(self.vocab)
        self.start_token = kwargs['start_token']
        self.end_token = kwargs['end_token']
        self.unk_token = kwargs['unk_token']
        self.characters = kwargs['characters']
        self.embed_dim = kwargs['embedding_size']
        self.hid_dim = kwargs['hidden_size']
        self.n_layers = kwargs['num_layers']
        self.nonlinearity = kwargs['nonlinearity']
        
        #Define the embedding layer
        self.embed = nn.Embedding(self.in_dim+1,self.embed_dim,padding_idx=self.in_dim)
        #Define the RNN layer
        self.rnn= nn.RNN(input_size=self.embed_dim,hidden_size=self.hid_dim,num_layers=self.n_layers,nonlinearity=self.nonlinearity.lower())
        #Define the output layer
        self.linear = nn.Linear(self.hid_dim,self.in_dim)
        #Define the softmax layer
        self.softmax = nn.LogSoftmax()
        
    def init_weights(self):
        #Randomly initialise all parameters
        torch.nn.init.xavier_uniform_(self.embed.weight)
        for i in range(self.n_layers):
            torch.nn.init.xavier_uniform_(getattr(self.rnn,'weight_hh_l'+str(i)))
            torch.nn.init.xavier_uniform_(getattr(self.rnn,'weight_ih_l'+str(i)))
            torch.nn.init.uniform_(getattr(self.rnn,'bias_hh_l'+str(i)))
            torch.nn.init.uniform_(getattr(self.rnn,'bias_ih_l'+str(i)))
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.uniform_(self.linear.bias)
    
    def forward(self, inputs, lengths):
        #Inputs are size (LxBx1)
        #Forward embedding layer
        emb = self.embed(inputs)
        #Embeddings are size (LxBxself.embed_dim)

        #Pack the sequences for RNN
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths)
        #Forward the LSTM
        packed_rec, self.hidden = self.rnn(packed,self.hidden)
        #Unpack the sequences
        rec, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_rec)
        #Hidden outputs are (LxBxself.hidden_size)
        
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
        #Reduce the batch size in the hidden state
        self.hidden=self.hidden[:,0:valid,:].contiguous()


