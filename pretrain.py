import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import time
import datetime

import scipy

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

parser.add_argument('--image_size', type=int, default=256, help='Image size. 64 for every experiment in the paper')

parser.add_argument('--update_interval', type=int, default=3, help='')
parser.add_argument('--log_interval', type=int, default=50, help='Print loss values every log_interval iterations.')
parser.add_argument('--image_save_interval', type=int, default=1000,
                    help='Save test results every image_save_interval iterations.')
parser.add_argument('--model_save_interval', type=int, default=10000,
                    help='Save models every model_save_interval iterations.')

parser.add_argument('--model_path', type=str, default='./models/',
                    help='path for saving trained models')

parser.add_argument('--crop_size', type=int, default=128, #224
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

parser.add_argument('--criterion', type=str, default='MSE')

parser.add_argument('--common_emb_size', type=int, default = 100)

def main():
    # global args
    args = parser.parse_args()

    now = datetime.datetime.now()
    current_date = now.strftime("%m-%d-%H-%M")

    assert args.criterion in ("MSE","Cosine","Hinge"), 'Invalid Loss Function'

    mask = args.common_emb_size
    assert mask <= args.hidden_size

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
    print("Loading Vocabulary...")
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
    decoder_Txt = TextDecoderOld(glove_emb, num_layers=1, bidirectional=False, hidden_size=args.hidden_size)
    # decoder_Txt = TextDecoder(encoder_Txt, glove_emb)
    # decoder_Txt = DecoderRNN(glove_emb, hidden_size=args.hidden_size)


    encoder_Img = ImageEncoder(img_dimension=args.crop_size,feature_dimension= args.hidden_size)
    decoder_Img = ImageDecoder(img_dimension=args.crop_size, feature_dimension= args.hidden_size)

    if cuda:
        encoder_Txt = encoder_Txt.cuda()
        decoder_Img = decoder_Img.cuda()

        encoder_Img = encoder_Img.cuda()
        decoder_Txt = decoder_Txt.cuda()


    # Losses and Optimizers
    print("Setting up the Objective Functions...")
    img_criterion = nn.MSELoss()
    # txt_criterion = nn.MSELoss(size_average=True)
    if args.criterion == 'MSE':
        txt_criterion = nn.MSELoss()
        cm_criterion = nn.MSELoss()
    elif args.criterion == "Cosine":
        txt_criterion = nn.CosineEmbeddingLoss(size_average=False)
        cm_criterion = nn.CosineEmbeddingLoss()
    else:
        txt_criterion = nn.HingeEmbeddingLoss(size_average=False)
        cm_criterion = nn.HingeEmbeddingLoss()


    if cuda:
        img_criterion = img_criterion.cuda()
        txt_criterion = txt_criterion.cuda()
        cm_criterion = cm_criterion.cuda()
    # txt_criterion = nn.CrossEntropyLoss()

    #     gen_params = chain(generator_A.parameters(), generator_B.parameters())
    print("Setting up the Optimizers...")
    # img_params = chain(decoder_Img.parameters(), encoder_Img.parameters())
    # txt_params = chain(decoder_Txt.decoder.parameters(), encoder_Txt.encoder.parameters())
    img_params = list(decoder_Img.parameters()) + list(encoder_Img.parameters())
    txt_params = list(decoder_Txt.decoder.parameters()) + list(encoder_Txt.encoder.parameters())

    # ATTENTION: Check betas and weight decay
    # ATTENTION: Check why valid_params fails on image networks with out of memory error

    img_optim = optim.Adam(img_params, lr=args.learning_rate,betas=(0.5, 0.999), weight_decay=0.00001)
    txt_optim = optim.Adam(valid_params(txt_params), lr=args.learning_rate,betas=(0.5, 0.999), weight_decay=0.00001)
    # img_enc_optim = optim.Adam(encoder_Img.parameters(), lr=args.learning_rate)#betas=(0.5, 0.999), weight_decay=0.00001)
    # img_dec_optim = optim.Adam(decoder_Img.parameters(), lr=args.learning_rate)#betas=(0.5,0.999), weight_decay=0.00001)
    # txt_enc_optim = optim.Adam(valid_params(encoder_Txt.encoder.parameters()), lr=args.learning_rate)#betas=(0.5,0.999), weight_decay=0.00001)
    # txt_dec_optim = optim.Adam(valid_params(decoder_Txt.decoder.parameters()), lr=args.learning_rate)#betas=(0.5,0.999), weight_decay=0.00001)


    train_images = False # Reverse 2
    for epoch in range(args.num_epochs):

        # TRAINING TIME
        print('EPOCH ::: TRAINING ::: ' + str(epoch + 1))
        batch_time = AverageMeter()
        txt_losses = AverageMeter()
        img_losses = AverageMeter()
        cm_losses = AverageMeter()
        end = time.time()

        bar = Bar('Training Net', max=len(data_loader))

        # Set training mode
        encoder_Img.train()
        decoder_Img.train()

        encoder_Txt.encoder.train()
        decoder_Txt.decoder.train()


        train_images = not train_images
        for i, (images, captions, lengths) in enumerate(data_loader):
            # ATTENTION REMOVE
            if i == 6450:
                break

            # Set mini-batch dataset
            images = to_var(images)
            captions = to_var(captions)

            # target = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # captions, lengths = pad_sequences(captions, lengths)
            # images = torch.FloatTensor(images)

            captions = captions.transpose(0,1).unsqueeze(2)
            lengths = torch.LongTensor(lengths)            # print(captions.size())


            # Forward, Backward and Optimize
            # img_optim.zero_grad()
            # img_dec_optim.zero_grad()
            # img_enc_optim.zero_grad()
            encoder_Img.zero_grad()
            decoder_Img.zero_grad()

            # txt_params.zero_grad()
            # txt_dec_optim.zero_grad()
            # txt_enc_optim.zero_grad()
            encoder_Txt.encoder.zero_grad()
            decoder_Txt.decoder.zero_grad()

            # Image Auto_Encoder Forward

            img_encoder_outputs, Iz  = encoder_Img(images)

            IzI = decoder_Img(img_encoder_outputs)

            img_rc_loss = img_criterion(IzI,images)


            # Text Auto Encoder Forward

            # target = target[:-1] # exclude last target from inputs

            captions = captions[:-1,:,:]
            lengths = lengths - 1
            dec_state = None

            encoder_outputs, memory_bank = encoder_Txt(captions, lengths)

            enc_state = \
                decoder_Txt.decoder.init_decoder_state(captions, memory_bank, encoder_outputs)

            decoder_outputs, dec_state, attns = \
                decoder_Txt.decoder(captions,
                             memory_bank,
                             enc_state if dec_state is None
                             else dec_state,
                             memory_lengths=lengths)

            Tz = encoder_outputs
            TzT = decoder_outputs


            # print("Decoder Output" ,TzT.size())
            # print("Captions: ",glove_emb(captions).size())

            if args.criterion == 'MSE':
                txt_rc_loss = txt_criterion(TzT,glove_emb(captions))
            else:
                txt_rc_loss = txt_criterion(TzT, glove_emb(captions),\
                                            Variable(torch.ones(TzT.size(0,1))).cuda())
            #
            # for x,y,l in zip(TzT.transpose(0,1),glove_emb(captions).transpose(0,1),lengths):
            #     if args.criterion == 'MSE':
            #         # ATTENTION dunno what's the right one
            #         txt_rc_loss += txt_criterion(x,y)
            #     else:
            #         # ATTENTION Fails on last batch
            #         txt_rc_loss += txt_criterion(x, y, Variable(torch.ones(x.size(0))).cuda())/l
            #
            # txt_rc_loss /= captions.size(1)



            # Computes Cross-Modal Loss

            Tz = Tz[0]

            txt =  Tz.narrow(1,0,mask)
            im = Iz.narrow(1,0,mask)

            if args.criterion == 'MSE':
                # cm_loss = cm_criterion(Tz.narrow(1,0,mask), Iz.narrow(1,0,mask))
                cm_loss = mse_loss(txt, im)
            else:
                cm_loss = cm_criterion(txt, im, \
                                       Variable(torch.ones(im.size(0)).cuda()))

            # K - Negative Samples
            k=5
            for _ in range(k):

                if cuda:
                    perm = torch.randperm(args.batch_size).cuda()
                else:
                    perm = torch.randperm(args.batch_size)


                sim  = (F.cosine_similarity(txt,txt[perm]) - 0.5)/2

                if args.criterion == 'MSE':
                    # cm_loss = cm_criterion(Tz.narrow(1,0,mask), Iz.narrow(1,0,mask))
                    cm_loss += mse_loss(txt, im[perm], sim)
                else:
                    cm_loss += sim * cm_criterion(txt, im[perm], \
                                           Variable(torch.ones(Tz.narrow(1,0,mask).size(0)).cuda()))




            # Computes the loss to be back-propagated
            rate = 0.2
            img_loss = img_rc_loss * (1 - rate) + cm_loss * rate
            txt_loss = txt_rc_loss * (1 - rate) + cm_loss * rate
            # txt_loss = txt_rc_loss + cm_loss
            # img_loss = img_rc_loss + cm_loss

            txt_losses.update(txt_rc_loss.data[0],args.batch_size)
            img_losses.update(img_rc_loss.data[0],args.batch_size)
            cm_losses.update(cm_loss.data[0], args.batch_size)

            # Half of the times we update one pipeline the others the other one
            if train_images:
                # Image Network Training and Backpropagation

                img_loss.backward()
                img_optim.step()
                # img_dec_optim.step()
                # img_enc_optim.step()

            else:
                # Text Nextwork Training & Back Propagation

                txt_loss.backward()
                txt_optim.step()
                # txt_dec_optim.step()
                # txt_enc_optim.step()


            if i % args.image_save_interval == 0:
                subdir_path = os.path.join( result_path, str(i / args.image_save_interval) )

                if os.path.exists( subdir_path ):
                    pass
                else:
                    os.makedirs( subdir_path )

                for im_idx in range(3):
                    im = (IzI[im_idx].cpu().data.numpy().transpose(1,2,0)/2+.5)*255
                    # im = (images[im_idx].cpu().data.numpy().transpose(1,2,0)+1.)*127.
                    filename_prefix = os.path.join (subdir_path, str(im_idx))
                    scipy.misc.imsave( filename_prefix + '.A.jpg', im)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_Img: {img_l:.3f}| Loss_Txt: {txt_l:.3f} | Loss_CM: {cm_l:.4f}'.format(
                        batch=i,
                        size=len(data_loader),
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        img_l=img_losses.avg,
                        txt_l=txt_losses.avg,
                        cm_l=cm_losses.avg,
                        )
            bar.next()
        bar.finish()

        # Save the models
        print('\n')
        print('Saving the models in {}...'.format(model_path))
        torch.save(decoder_Img.state_dict(),
                   os.path.join(model_path,
                                'decoder-img-%d-' %(epoch+1)) + current_date + ".pkl")
        torch.save(encoder_Img.state_dict(),
                   os.path.join(model_path,
                                'encoder-img-%d-' %(epoch+1)) + current_date + ".pkl")
        torch.save(decoder_Txt.state_dict(),
                   os.path.join(model_path,
                                'decoder-txt-%d-' %(epoch+1)) + current_date + ".pkl")
        torch.save(encoder_Txt.state_dict(),
                   os.path.join(model_path,
                                'encoder-txt-%d-' %(epoch+1)) + current_date + ".pkl")

        #
        # # VALIDATION TIME
        # print('EPOCH ::: VALIDATION ::: ' + str(epoch + 1))
        #
        # # Set Evaluation Mode
        # encoder_Img.eval()
        #
        # encoder_Txt.encoder.eval()
        #
        # # get pairs
        # end = time.time()
        #
        # bar = Bar('Computing Validation Set Embeddings', max=len(val_loader))
        #
        # for i, (images, captions, lengths) in enumerate(val_loader):
        #
        #     # Set mini-batch dataset
        #     images = to_var(images)
        #     captions = to_var(captions)
        #
        #     captions = captions.transpose(0,1).unsqueeze(2)
        #     lengths = torch.LongTensor(lengths)
        #
        #     _, img_emb = encoder_Img(images)
        #
        #     txt_emb, _ = encoder_Txt(captions, lengths)
        #
        #
        #     # current_embeddings = torch.cat( \
        #     #         (txt_emb.transpose(0,1).data,img_emb.unsqueeze(1).data)
        #     #         , 1)
        #     current_embeddings = np.concatenate( \
        #         (txt_emb.cpu().data.numpy(),\
        #          img_emb.unsqueeze(0).cpu().data.numpy())\
        #         ,0)
        #
        #     # current_embeddings = img_emb.data
        #     if i:
        #         # result_embeddings = torch.cat( \
        #         result_embeddings = np.concatenate( \
        #             (result_embeddings, current_embeddings) \
        #             ,1)
        #     else:
        #         result_embeddings = current_embeddings
        #
        #     # measure elapsed time
        #     batch_time.update(time.time() - end)
        #     end = time.time()
        #
        #     # plot progress
        #     bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
        #                 batch=i,
        #                 size=len(val_loader),
        #                 bt=batch_time.avg,
        #                 total=bar.elapsed_td,
        #                 eta=bar.eta_td,
        #                 )
        #     bar.next()
        # bar.finish()
        #
        #
        # print("Computing Nearest Neighbors...")
        # print("Computing Nearest Neighbors...")
        #
        # neigh = NearestNeighbors(5)
        # neigh.fit(result_embeddings[1])
        # kneigh = neigh.kneighbors(result_embeddings[0], return_distance=False)
        # top5 = np.mean([i in nn for i,nn in enumerate(kneigh)])
        #
        # print("Top-5 accuracy for Image Retrieval: ",top5)










if __name__ == "__main__":
    main()
