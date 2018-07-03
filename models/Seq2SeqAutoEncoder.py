from onmt import Models
import torch
import torch.nn as nn
import torch.functional as F

class TextEncoder(nn.Module):
    def __init__(
            self,
            embedding,
            hidden_size = 300,
            num_layers = 1,
            bidirectional = True,
            ):
        super(TextEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers= num_layers, bidirectional= bidirectional)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size).cuda

class TextDecoder(nn.Module):
    def __init__(
            self,
            embeddings,
            output_size,
            num_layers = 1,
            hidden_size = 300,
            bidirectional = False
            ):
        super(TextDecoder,self).__init__()

        self.hidden_size = hidden_size

        self.embedding = embeddings
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers= num_layers, bidirectional = bidirectional)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_sized)


