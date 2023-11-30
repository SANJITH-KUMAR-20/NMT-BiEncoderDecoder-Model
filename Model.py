import torch
import torch.nn as nn
import os
import shutil


class Encoder(nn.Module): # class for the encoder model

    def __init__(self,input_shape,hidden_layers,bidirectional = True,no_of_layers = 1,keep_prob = 0.8,*args,
                 **kwargs):
        super(Encoder,self).__init__(*args,**kwargs)

        self.input_shape = input_shape
        self.hidden_layers = hidden_layers
        self.no_of_layers = no_of_layers
        self.bidirectional = bidirectional
        self.dropout_value = 1-keep_prob
        self.dropout = nn.Dropout(self.dropout_value)
        self.encoder_embeddings = nn.Embedding(input_shape,hidden_layers)
        self.Lstm = nn.LSTM(input_size = hidden_layers,hidden_size = hidden_layers,
                            num_layers = no_of_layers,dropout = 1-keep_prob,bidirectional = bidirectional,batch_first = False)
        if not bidirectional:
            self.fully_connected_layer = nn.Linear(hidden_layers,hidden_layers)
        else:
            self.fully_connected_layer = nn.Linear(hidden_layers*2,hidden_layers)

    def forward(self,input_data,h_hidden,c_hidden,lstm = True): # forward pass
        embeddings = self.encoder_embeddings(input_data)
        drop = self.dropout(embeddings)
        if lstm:
            h_,outputs = self.Lstm(drop,(h_hidden,c_hidden))
        else:
            h_,outputs = self.Gru(drop,(h_hidden,c_hidden))

        return h_,outputs
    
    def initialize_weights(self,batch_size): # Initializing the weights
        if not self.bidirectional:
            h_hidden = torch.autograd.Variable(torch.zeros(self.no_of_layers,batch_size,self.hidden_layers))
            c_hidden = torch.autograd.Variable(torch.zeros(self.no_of_layers,batch_size,self.hidden_layers))
        else:
            h_hidden = torch.autograd.Variable(torch.zeros(self.no_of_layers*2,batch_size,self.hidden_layers))
            c_hidden = torch.autograd.Variable(torch.zeros(self.no_of_layers*2,batch_size,self.hidden_layers))
        if torch.cuda.is_available():
            return h_hidden.cuda(),c_hidden.cuda()
        else:
            return h_hidden,c_hidden
        

class Decoder(nn.Module):

    def __init__(self,hidden_shape,output_shape,no_of_layers = 1,keep_prob = 0.8,
                 bidirectional=True,*args,**kwargs):
        super(Decoder,self).__init__(*args,**kwargs)
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.bidirectional = bidirectional
        self.no_of_layers = no_of_layers
        self.dropout_value = 1-keep_prob
        self.dropout = nn.Dropout(1-keep_prob)
        self.embeddings = nn.Embedding(output_shape,hidden_shape)
        self.Lstm = nn.LSTM(input_size = hidden_shape,hidden_size = hidden_shape,num_layers = no_of_layers,
                            dropout = 1-keep_prob,bidirectional=bidirectional,batch_first = False)
        if not bidirectional:
            self.Inter = nn.Linear(hidden_shape,hidden_shape)
            self.Context_comb = nn.Linear(hidden_shape+hidden_shape,hidden_shape)
        else:
            self.Inter = nn.Linear(hidden_shape*2,hidden_shape*2)
            self.Context_comb = nn.Linear(hidden_shape*2+hidden_shape*2,hidden_shape)
        self.output = nn.Linear(hidden_shape,output_shape)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self,input_data,h_hidden,c_hidden,encoder_outputs):
        embeddings = self.embeddings(input_data)
        embeddings = self.dropout(embeddings)
        batch_size = embeddings.shape[1]
        hiddens,outputs = self.Lstm(embeddings,(h_hidden,c_hidden))

        hid = outputs[0].view(self.no_of_layers,1 if not self.bidirectional else 2, hiddens.shape[1],self.hidden_shape)[self.no_of_layers-1]
        hid = hid.permute(1,2,0).contiguous().view(batch_size,-1,1)

        lr = self.Inter(encoder_outputs.permute(1,0,2))
        scores = torch.bmm(lr,hid)
        attn = self.softmax(scores)
        conmat = torch.bmm(encoder_outputs.permute(1,2,0),attn)
        h_tilde = self.tanh(self.Context_comb(torch.cat((conmat,hid),dim=1).view(batch_size,-1)))
        pred = self.output(h_tilde)
        pred = self.log_softmax(pred)
        return pred, outputs