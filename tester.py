import onmt.Models as models
import onmt.ModelConstructor as con

import onmt

import opts

import json
import os

import torchvision.datasets as dset
import torchvision.transforms as transforms

import torchtext.data as data

import spacy
spacy_en = spacy.load('en')

def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]

# cap = dset.CocoCaptions(root = '/media/ctrlaltv/USBitch/MSCOCO/val2014',
#                         annFile = '/media/ctrlaltv/USBitch/MSCOCO/annotations/captions_val2014.json',
#                         transform=transforms.ToTensor())
2
# print(len(cap))
# for i,(img, txt) in enumerate(cap):

# annotations = "/media/ctrlaltv/USBitch/MSCOCO/annotations/"
annotations = "annotations/"
val = json.load(open(annotations + 'captions_val2014.json', 'r'))
train = json.load(open(annotations + 'captions_train2014.json', 'r'))

# val = data.TabularDataset(
#        path=(annotations + 'captions_val2014.json'), format='json',
#        fields={'annotations': ('caption',  data.Field(sequential=True, tokenize=tokenizer, lower=True))})

vocab = set([])
with open("data/valid.src.txt", 'wb') as f:
    for annot in val["annotations"]:
        cap = dict(annot)["caption"]
    # vocab.update(set(cap.lower().rstrip("\n").rstrip(".").rstrip(",").split(" ")))
        f.write(" ".join(tokenizer(cap)) + "\n")

with open("data/train.src.txt", 'wb') as f:
    for annot in train["annotations"]:
        cap = dict(annot)["caption"]
        f.writelines(" ".join(tokenizer(cap)) + "\n")
    # vocab.update(set(cap.lower().rstrip("\n").rstrip(".").rstrip(",").split(" ")))

    # vocab.update(set(tokenizer(cap)))

# vocab_size = len(vocab)
# word_to_ix = {word: i for i, word in enumerate(vocab)}
#




# for cap in val.fields.iteritems():
#        print(cap)



