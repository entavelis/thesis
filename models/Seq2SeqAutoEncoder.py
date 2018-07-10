from onmt import Models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class TextEncoder(nn.Module):
    def __init__(
            self,
            embedding,
            hidden_size = 300,
            num_layers = 1,
            bidirectional = True,
            model = "GRU"
            ):
        super(TextEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        if model == "GRU":
            self.rnn = nn.GRU(embedding.embedding_dim, hidden_size, num_layers= num_layers, bidirectional= bidirectional, \
                          batch_first= True)
        else:
            self.rnn = nn.LSTM(embedding.embedding_dim, hidden_size, num_layers= num_layers, bidirectional= bidirectional, \
                               batch_first= True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return Variable(torch.zeros(1,batch_size, self.hidden_size).cuda())

class TextDecoder(nn.Module):
    def __init__(
            self,
            embeddings,
            output_size,
            num_layers = 1,
            hidden_size = 300,
            bidirectional = False,
            model = "GRU"
            ):
        super(TextDecoder,self).__init__()

        self.hidden_size = hidden_size

        self.embedding = embeddings
        if model == "GRU":
            self.rnn = nn.GRU(embeddings.embedding_dim, hidden_size, num_layers= num_layers, bidirectional = bidirectional, \
                          batch_first= True)
        else:
            self.rnn = nn.LSTM(embeddings.embedding_dim, hidden_size, num_layers= num_layers, bidirectional = bidirectional, \
                          batch_first= True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input)
        # output = self.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.out(output[:,0])
        # output = self.softmax(self.out(output))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size).cuda()


