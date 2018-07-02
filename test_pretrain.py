import os
import time

import argparse
from itertools import chain

import onmt
from onmt.modules import Embeddings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torchtext import vocab
from model import *
import pickle

from utils import *

import scipy

from image_caption.build_vocab import Vocabulary
from image_caption.data_loader import get_loader

from pytorch_classification.utils import Bar, AverageMeter

from sklearn.neighbors import NearestNeighbors



# from progressbar import ETA, Bar, Percentage, ProgressBar

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')
parser.add_argument('--epoch_size', type=int, default=5000, help='Set epoch size')
parser.add_argument('--result_path', type=str, default='./results/',
                    help='Set the path the result images will be saved.')

parser.add_argument('--image_size', type=int, default=70, help='Image size. 64 for every experiment in the paper')

parser.add_argument('--update_interval', type=int, default=3, help='')
parser.add_argument('--log_interval', type=int, default=50, help='Print loss values every log_interval iterations.')
parser.add_argument('--image_save_interval', type=int, default=1000,
                    help='Save test results every image_save_interval iterations.')
parser.add_argument('--model_save_interval', type=int, default=10000,
                    help='Save models every model_save_interval iterations.')

parser.add_argument('--model_path', type=str, default='./models/',
                    help='path for saving trained models')

parser.add_argument('--crop_size', type=int, default=64, #224
                    help='size for randomly cropping images')
parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                    help='path for vocabulary wrapper')
parser.add_argument('--image_dir', type=str, default='./data/resized2014',
                    help='directory for resized images')
parser.add_argument('--valid_dir', type=str, default='./data/val_resized2014',
                    help='directory for resized validation set images')

parser.add_argument('--embedding_path', type=str,
                    default='./glove/',
                    help='path for pretrained embeddings')
parser.add_argument('--caption_path', type=str,
                    default='./data/annotations/captions_train2014.json',
                    help='path for train annotation json file')
parser.add_argument('--valid_caption_path', type=str,
                    default='./data/annotations/captions_val2014.json',
                    help='path for valid annotation json file')
parser.add_argument('--log_step', type=int, default=10,
                    help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=1000,
                    help='step size for saving trained models')


# Model parameters
parser.add_argument('--embedding_size', type=int, default=300)
parser.add_argument('--hidden_size', type=int, default=300,
                    help='dimension of lstm hidden states')
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in lstm')

parser.add_argument('--extra_layers', type=str, default='true')
parser.add_argument('--fixed_embeddings', type=str, default="true")
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.001)

parser.add_argument('--criterion', type=str, default='Cosine')

parser.add_argument('--common_emb_size', type=int, default = 100)

def main():
    # global args
    args = parser.parse_args()

    assert args.criterion in ("MSE","Cosine","Hinge"), 'Invalid Loss Function'

    cuda = args.cuda
    if cuda == 'true':
        cuda = True
    else:
        cuda = False

    # Image preprocessing //ATTENTION
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    result_path = args.result_path
    model_path = args.model_path

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)


    # Load vocabulary wrapper.
    print('\n')
    print("\033[94mLoading Vocabulary...")
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Load Embeddings
    emb_size = args.embedding_size
    emb_path = args.embedding_path
    if args.embedding_path[-1]=='/':
        emb_path += 'glove.6B.' + str(emb_size) + 'd.txt'

    print("Loading Embeddings...")
    emb = load_glove_embeddings(emb_path, vocab.word2idx, emb_size)

    glove_emb = Embeddings(emb_size,len(vocab.word2idx),vocab.word2idx["<pad>"])
    glove_emb.word_lut.weight.data.copy_(emb)
    glove_emb.word_lut.weight.requires_grad = False

    # glove_emb = nn.Embedding(emb.size(0), emb.size(1))
    # glove_emb = embedding(emb.size(0), emb.size(1))
    # glove_emb.weight = nn.Parameter(emb)


    # Freeze weighs
    # if args.fixed_embeddings == "true":
        # glove_emb.weight.requires_grad = False

    # Build data loader
    print("Building Data Loader For Test Set...")
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,

                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    print("Building Data Loader For Validation Set...")
    val_loader = get_loader(args.valid_dir, args.valid_caption_path, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    print("Setting up the Networks...")
    encoder_Txt = TextEncoderOld(glove_emb, num_layers=1, bidirectional=False, hidden_size=args.hidden_size)
    # decoder_Txt = TextDecoderOld(glove_emb, num_layers=1, bidirectional=False, hidden_size=args.hidden_size)
    # decoder_Txt = TextDecoder(encoder_Txt, glove_emb)
    # decoder_Txt = DecoderRNN(glove_emb, hidden_size=args.hidden_size)


    encoder_Img = ImageEncoder(img_dimension=args.crop_size,feature_dimension= args.hidden_size)
    # decoder_Img = ImageDecoder(img_dimension=args.crop_size, feature_dimension= args.hidden_size)

    if cuda:
        encoder_Txt = encoder_Txt.cuda()

        encoder_Img = encoder_Img.cuda()


    for epoch in range(args.num_epochs):


        # VALIDATION TIME
        print('\033[92mEPOCH ::: VALIDATION ::: ' + str(epoch + 1))

        # Load the models
        print("Loading the models...")

        # suffix = '-{}-05-28-13-14.pkl'.format(epoch+1)
        # mask = 300

        prefix = ""
        suffix = '-{}-05-28-09-23.pkl'.format(epoch+1)
        # suffix = '-{}-05-28-11-35.pkl'.format(epoch+1)
        # suffix = '-{}-05-28-16-45.pkl'.format(epoch+1)
        # suffix = '-{}-05-29-00-28.pkl'.format(epoch+1)
        # suffix = '-{}-05-29-00-30.pkl'.format(epoch+1)
        # suffix = '-{}-05-29-01-08.pkl'.format(epoch+1)
        mask = 200

        # suffix = '-{}-05-28-15-39.pkl'.format(epoch+1)
        # suffix = '-{}-05-29-12-11.pkl'.format(epoch+1)
        # suffix = '-{}-05-29-12-14.pkl'.format(epoch+1)
        # suffix = '-{}-05-29-14-24.pkl'.format(epoch+1) #best
        # suffix = '-{}-05-29-15-43.pkl'.format(epoch+1)
        date = "06-30-14-22"
        date = "07-01-12-49" #bad
        date = "07-01-16-38"
        date = "07-01-18-16"
        date = "07-02-15-38"
        prefix = "{}/".format(date)
        suffix = '-{}-{}.pkl'.format(epoch+1,date)
        mask = 100

        print(suffix)
        try:
            encoder_Img.load_state_dict(torch.load(os.path.join(args.model_path,
                                    prefix + 'encoder-img' + suffix)))
            encoder_Txt.load_state_dict(torch.load(os.path.join(args.model_path,
                                    prefix + 'encoder-txt' + suffix)))
        except FileNotFoundError:
            print("\n\033[91mFile not found...\nTerminating Validation Procedure!")
            break


        # Set Evaluation Mode
        encoder_Img.eval()

        encoder_Txt.encoder.eval()

        batch_time = AverageMeter()
        end = time.time()

        loader = val_loader
        bar = Bar('Computing Validation Set Embeddings', max=len(loader))

        limit = 12
        for i, (images, captions, lengths) in enumerate(loader):
            if i == limit:
                break

            # Set mini-batch dataset
            images = to_var(images)
            captions = to_var(captions)

            captions = captions.transpose(0,1).unsqueeze(2)
            lengths = torch.LongTensor(lengths)

            _, img_emb = encoder_Img(images)

            txt_emb, _ = encoder_Txt(captions, lengths)

            img_emb = img_emb.narrow(1,0,mask)
            txt_emb = txt_emb.narrow(2,0,mask)
            # current_embeddings = torch.cat( \
            #         (txt_emb.transpose(0,1).data,img_emb.unsqueeze(1).data)
            #         , 1)
            current_embeddings = np.concatenate( \
                (txt_emb.cpu().data.numpy(),\
                 img_emb.unsqueeze(0).cpu().data.numpy())\
                ,0)

            # current_embeddings = img_emb.data
            if i:
                # result_embeddings = torch.cat( \
                result_embeddings = np.concatenate( \
                    (result_embeddings, current_embeddings) \
                    ,1)
            else:
                result_embeddings = current_embeddings

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                        batch=i,
                        size=len(val_loader),
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        )
            bar.next()
        bar.finish()


        a = [((result_embeddings[0][i] - result_embeddings[1][i]) ** 2).mean() for i in range(limit*args.batch_size)]
        print("Validation MSE: ",np.mean(a))
        print("Validation MSE: ",np.mean(a))

        print("Computing Nearest Neighbors...")
        i = 0
        topk = []
        kss = [1,10,50]
        for k in kss:

            if i:
                print("Normalized ")
                result_embeddings[0] = result_embeddings[0]/result_embeddings[0].sum()
                result_embeddings[1] = result_embeddings[1]/result_embeddings[1].sum()

            # k = 5
            neighbors = NearestNeighbors(k, metric = 'cosine')
            neigh = neighbors
            neigh.fit(result_embeddings[1])
            kneigh = neigh.kneighbors(result_embeddings[0], return_distance=False)

            ks = set()
            for n in kneigh:
                ks.update(set(n))

            print(len(ks)/result_embeddings.shape[1])

            # a = [((result_embeddings[0][i] - result_embeddings[1][i]) ** 2).mean() for i in range(128)]
            # rs = result_embeddings.sum(2)
            # a = (((result_embeddings[0][0]- result_embeddings[1][0])**2).mean())
            # b = (((result_embeddings[0][0]- result_embeddings[0][34])**2).mean())
            topk.append(np.mean([int(i in nn) for i,nn in enumerate(kneigh)]))

        print("Top-{k:},{k2:},{k3:} accuracy for Image Retrieval:\n\n\t\033[95m {tpk: .3f}% \t {tpk2: .3f}% \t {tpk3: .3f}% \n".format(
                      k=kss[0],
                      k2=kss[1],
                      k3=kss[2],
                      tpk= 100*topk[0],
                      tpk2= 100*topk[1],
                      tpk3= 100*topk[2]))





if __name__ == "__main__":
    main()
