# Based on https://github.com/amirbar/rnn.wgan/blob/master/model.py
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import gumbel_softmax
from utils import add_gaussian


class SeqDiscriminator(nn.Module):

    def __init__(self,  embedding, vocab_size,
                hidden_size = 300,
                num_layers=1,
                sos_idx = 1,
                eos_idx = 2,
                pad_idx = 0,
                max_sequence_length = 30,
                batch_size = 128,
                word_dropout = 0.5,
                bidirectional = False,
                mask = 0.5,
                latent_size= 512):

        super().__init__()

        self.masked_size = int(mask*latent_size)
        self.img_embedding = nn.Linear(self.masked_size,128)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.word_dropout = nn.Dropout(p=word_dropout)

        embedding_size = embedding.embedding_dim

        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(embedding_size + 128,1)

    def forward(self,input, img_emb, lengths):

        if len(input.size()) > 2:
            lengths, sorted_idx = torch.sort(lengths, descending=True)
            input = input[sorted_idx]
            img_emb = img_emb[sorted_idx]

            gumbel = torch.zeros_like(input)
            for i in range(input.size(1)):
                gumbel[:,i] = gumbel_softmax(input[:,i].squeeze(), tau=0.5).unsqueeze(1)

            input_emb = torch.mm(input.view(-1,input.size(-1)), self.embedding.weight)\
                .view(input.size(0),-1, self.embedding.embedding_dim)
        else:
            input_emb = self.embedding(input)
            # input_emb = add_gaussian(input_emb, std=0.01)

        input_emb = self.word_dropout(input_emb)
        packed_input = rnn_utils.pack_padded_sequence(input_emb, lengths.data.tolist(), batch_first=True)

        output, hidden = self.rnn(packed_input)

        img = self.img_embedding(img_emb[:,:self.masked_size])
        # augmented = torch.cat([output[:,-1,:].squeeze() ,img],1)
        augmented = torch.cat([hidden.squeeze() ,img],1)
        pred = self.fc(augmented)

        return torch.sigmoid(pred).mean(0), hidden # .view(1)




