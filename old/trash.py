### A repository of past ideas/functions


# class TextEncoderOld(nn.Module):
#     def __init__(
#             self,
#             embeddings
#             ):
#         super(TextEncoder, self).__init__()
#
#         self.hidden_dim = embedding_dim
#
#         rnn_type = "GRU"
#         brnn = True
#         enc_layers = 100
#         rnn_size = latent_dim
#         dropout = 0.3
#         bridge = True
#         self.encoder = Models.RNNEncoder(rnn_type, brnn, enc_layers,
#                           rnn_size, dropout, embeddings,
#                           bridge)
#
#     def forward(self, src, lengths):
#
#         # tgt = tgt[:-1]  # exclude last target from inputs
#         return self.encoder(src, lengths)
#
# class TextDecoderOld(nn.Module):
#     def __init__(
#             self,
#             embeddings
#             ):
#
#         rnn_type = "GRU"
#         brnn = True
#         dec_layers =2
#         rnn_size = latent_dim
#         global_attention = True
#         coverage_attn = True
#         context_gate = True
#         copy_attn = True
#         dropout = 0.3
#         reuse_copy_attn = True
#
#
#         self.decoder = Models.StdRNNDecoder(rnn_type, brnn,
#                              dec_layers, rnn_size,
#                              global_attention,
#                              coverage_attn,
#                              context_gate,
#                              copy_attn,
#                              dropout,
#                              embeddings,
#                              reuse_copy_attn)

