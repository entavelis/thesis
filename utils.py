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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
        # y_hard = logits.new_zeros(*shape).scatter_(-1, k.view(-1, 1), 1.0)
        y_hard = torch.zeros(*shape).cuda().scatter_(-1, k.view(-1, 1).data, 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y

def add_gaussian(input, std = 0.05):
    input = input + Variable(input.data.new(input.size()).normal_(0,std))
    return input

def get_gen_lengths(input, eos_idx = 2):
    gen_lens = to_var(torch.zeros(input.size(0)).long())
    not_found = to_var(torch.ones(input.size(0)).byte())
    for word_i in range(input.size(1)):
        _, maxarg = torch.topk(input[:,word_i,:], 1, -1)
        gen_lens += not_found.long()
        not_found = not_found & (maxarg.squeeze() != eos_idx)

    return gen_lens

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
    try:
        m.bias.data.fill_(0)
    except:
        pass

