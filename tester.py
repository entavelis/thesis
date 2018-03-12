import onmt.Models as models
import onmt.ModelConstructor as con

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

cap = dset.CocoCaptions(root = '/media/ctrlaltv/USBitch/MSCOCO/val2014',
                        annFile = '/media/ctrlaltv/USBitch/MSCOCO/annotations/captions_val2014.json',
                        transform=transforms.ToTensor())

print(len(cap))
# for i,(img, txt) in enumerate(cap):

# annotations = "/media/ctrlaltv/USBitch/MSCOCO/annotations/"
# val = json.load(open(annotations + 'captions_val2014.json', 'r'))
# train = json.load(open(annotations + 'captions_train2014.json', 'r'))

# val = data.TabularDataset(
#        path=(annotations + 'captions_val2014.json'), format='json',
#        fields={'annotations': ('caption',  data.Field(sequential=True, tokenize=tokenizer, lower=True))})

vocab = set([])
for annot in val["annotations"]:
    cap = dict(annot)["caption"]
    vocab.update(set(cap.lower().rstrip("\n").rstrip(".").rstrip(",").split(" ")))

# print(len(vocab))
# for cap in val.fields.iteritems():
#        print(cap)



