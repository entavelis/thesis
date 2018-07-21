from onmt import Models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class TextDecoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            output_size,
            emb_size = 300,
            num_layers = 1,
            hidden_size = 300,
            bidirectional = False,
            model = "GRU"
            ):
        super(TextDecoder,self).__init__()

        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(emb_size, hidden_size)
        if model == "GRU":
            self.rnn = nn.GRU(self.embeddings.embedding_dim, hidden_size, num_layers= num_layers, bidirectional = bidirectional, \
                          batch_first= True)
        else:
            self.rnn = nn.LSTM(self.embeddings.embedding_dim, hidden_size, num_layers= num_layers, bidirectional = bidirectional, \
                          batch_first= True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, max_len):

        for i in range(max_len):
            hiddens, states = self.rnn(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.out(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
              # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)


        inputs = inputs.unsqueeze(1)

def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size).cuda()

