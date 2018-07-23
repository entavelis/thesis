import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#source: https://github.com/A-Jacobson/CNN_Sentence_Classification/blob/master/WordVectors.ipynb
def load_glove_embeddings(path, word2idx, embedding_dim=100):
    print("Loading from path: " + path)
    with open(path,encoding='utf-8') as f:
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word2idx.get(word)
            if index:
                vector = np.array(values[1:], dtype='float32')
                embeddings[index] = vector
        return torch.from_numpy(embeddings).float()



def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def as_np(data):
    return data.cpu().data.numpy()

def pad_sequences(seqs,lens):
    seqs = sorted(seqs, key=lambda x: len(x), reverse=True)
    # lens = [len(seq) for seq in seqs]
    padded_seqs = torch.zeros(len(seqs), max(lens)).long()
    for i, seq in enumerate(seqs):
        end = lens[i]
        # Changed from Long to Float because: embeddings
        padded_seqs[i, :end] = torch.LongTensor(seq[:end])

    padded_seqs.transpose(0, 1)

    return padded_seqs, lens

def valid_params(params):
    return [p for p in params if p.requires_grad]

class embedding(nn.Embedding):


    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(embedding, self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                 sparse, _weight)

        self.embedding_size = embedding_dim

# class embedding(object):
#     def __init__(self, wrapped_class, *args, **kargs):
#         self.wrapped_class = wrapped_class(*args, **kargs)
#
#     def __getattr__(self,attr):
#         if attr=="embedding_size":
#             return self.wrapped_class.__getattribute__("embedding_dim")
#
#         orig_attr =  self.wrapped_class.__getattribute__(attr)
#         if callable(orig_attr):
#             return orig_attr(*args, **kwargs)
#         else:
#             return orig_attr



# From pytorch functional

def _sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    dims = logits.dim()
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps, out=logits.data.new())
    y = torch.add(logits, Variable(gumbel_noise))
    return F.softmax(y / tau, dims - 1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    r"""
    Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
      logits: `[batch_size, num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd

    Returns:
      Sampled tensor of shape ``batch_size x num_features`` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across features

    Constraints:

    - Currently only work on 2D input :attr:`logits` tensor of shape ``batch_size x num_features``

    Based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    shape = logits.size()
    assert len(shape) == 2
    y_soft = _gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        _, k = y_soft.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = logits.new_zeros(*shape).scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y
