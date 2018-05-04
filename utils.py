import numpy as np
import torch
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

def pad_sequences(seqs):
    seqs = sorted(seqs, key=lambda x: len(x), reverse=True)
    lens = [len(seq) for seq in seqs]
    padded_seqs = torch.zeros(len(seqs), max(lens)).long()
    for i, seq in enumerate(seqs):
        end = lens[i]
        # Changed from Long to Float because: embeddings
        padded_seqs[i, :end] = torch.LongTensor(seq[:end])

    padded_seqs.transpose(0, 1)
    return padded_seqs, lens

def valid_params(params):
    return [p for p in params if p.requires_grad]

