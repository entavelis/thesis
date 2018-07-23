# Based on https://github.com/amirbar/rnn.wgan/blob/master/model.py
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var


class SeqDiscriminator(nn.Module):

    def __init__(self,  embedding, vocab_size,
                hidden_size = 300,
                num_layers=1,
                sos_idx = 1,
                eos_idx = 2,
                pad_idx = 0,
                max_sequence_length = 30,
                batch_size = 128):

        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        embedding_size = embedding.embedding_dim

        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(embedding_size,1)

    def forward(self,input):

        input_emb = torch.mm(input.view(-1,input.size(-1)), self.embedding.weight)\
            .view(input.size(0),-1, self.embedding.embedding_dim)
        output, hidden = self.rnn(input_emb)
        pred = self.fc(output[:,-1,:])

        return torch.sigmoid(pred).mean(0) # .view(1)




