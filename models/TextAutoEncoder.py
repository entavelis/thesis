from onmt import Models
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(
            self,
            embedding,
            hidden_size = 300,
            num_layers = 1,
            bidirectional = True,
            bridge = False
            ):
        super(TextEncoder, self).__init__()

        self.hidden_dim = hidden_size
        self.num_layers = num_layers

        self.embedding = embedding
        self.enc_layers = self.embedding.embedding_size

        rnn_type = "GRU"
        brnn = bidirectional

        rnn_size = self.hidden_dim
        dropout = 0.3
        self.encoder = Models.RNNEncoder(rnn_type, brnn, num_layers,
                          rnn_size, dropout, embedding,
                          bridge)

    def forward(self, src, lengths):

        # tgt = tgt[:-1]  # exclude last target from inputs
        # src = src.transpose(0,1).unsqueeze(2)
        # lengths = torch.LongTensor(lengths)
        # print(len(lengths))
        return self.encoder(src, lengths)

class TextDecoder(nn.Module):
    def __init__(
            self,
            embeddings,
            num_layers = 1,
            rnn_type="GRU",
            hidden_size = 300,
            bidirectional = False
            ):
        super(TextDecoder,self).__init__()

        global_attention = "general"
        coverage_attn = False
        context_gate = None
        copy_attn = False
        reuse_copy_attn = False
        dropout = 0.3


        self.decoder = Models.StdRNNDecoder(rnn_type, bidirectional,
                             num_layers, hidden_size,
                             global_attention,
                             coverage_attn,
                             context_gate,
                             copy_attn,
                             dropout,
                             embeddings,
                             reuse_copy_attn)

        def forward(self, *input, **kargs):
            return self.decoder(*input, **kargs)

