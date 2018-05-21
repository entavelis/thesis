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

parser.add_argument('--criterion', type=str, default='Cosine')
def main():
    # global args
    args = parser.parse_args()

    assert args.criterion in ("L1","Cosine","Hinge"), 'Invalid Loss Function'

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
    #     generator_A = Generator()
    encoder_Txt = TextEncoderOld(glove_emb, bidirectional=False, hidden_size=args.hidden_size)
    # decoder_Txt = TextDecoder(encoder_Txt, glove_emb)
    # decoder_Txt = DecoderRNN(glove_emb, hidden_size=args.hidden_size)
    decoder_Txt = TextDecoderOld(glove_emb, hidden_size=args.hidden_size)



    #     generator_B = Generator()
    encoder_Img = ImageEncoder(img_dimension=args.crop_size,feature_dimension= args.hidden_size)
    decoder_Img = ImageDecoder(img_dimension=args.crop_size, feature_dimension= args.hidden_size)

    if cuda:
        # test_I = test_I.cuda()
        # test_T = test_T.cuda()

        #          generator_A = generator_A.cuda()
        #        generator_A = generator_A.cuda()

        encoder_Txt = encoder_Txt.cuda()
        decoder_Img = decoder_Img.cuda()

        #         generator_B = generator_B.cuda()
        encoder_Img = encoder_Img.cuda()
        decoder_Txt = decoder_Txt.cuda()


    # Losses and Optimizers
    print("Setting up the Objective Functions...")
    img_criterion = nn.MSELoss()
    # txt_criterion = nn.MSELoss(size_average=True)
    if args.criterion == 'L1':
        txt_criterion = nn.L1Loss()
        cm_criterion = nn.L1Loss()
    elif args.criterion == "Cosine":
        txt_criterion = nn.CosineEmbeddingLoss()
        cm_criterion = nn.CosineEmbeddingLoss()
    else:
        txt_criterion = nn.HingeEmbeddingLoss()
        cm_criterion = nn.HingeEmbeddingLoss()


    # txt_criterion = nn.CrossEntropyLoss()

    #     gen_params = chain(generator_A.parameters(), generator_B.parameters())
    print("Setting up the Optimizers...")
    # img_params = chain(decoder_Img.parameters(), encoder_Img.parameters())
    # txt_params = chain(decoder_Txt.parameters(), encoder_Txt.parameters())

    # ATTENTION: Check betas and weight decay
    # ATTENTION: Check why valid_params fails on image networks with out of memory error
    img_enc_optim = optim.Adam(encoder_Img.parameters(), lr=args.learning_rate)#betas=(0.5, 0.999), weight_decay=0.00001)
    img_dec_optim = optim.Adam(decoder_Img.parameters(), lr=args.learning_rate)#betas=(0.5,0.999), weight_decay=0.00001)
    txt_enc_optim = optim.Adam(valid_params(encoder_Txt.parameters()), lr=args.learning_rate)#betas=(0.5,0.999), weight_decay=0.00001)
    txt_dec_optim = optim.Adam(valid_params(decoder_Txt.parameters()), lr=args.learning_rate)#betas=(0.5,0.999), weight_decay=0.00001)


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


        for i, (images, captions, lengths) in enumerate(data_loader):

            # Set mini-batch dataset
            images = to_var(images)
            captions = to_var(captions)

            # target = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # captions, lengths = pad_sequences(captions, lengths)
            # images = torch.FloatTensor(images)

            captions = captions.transpose(0,1).unsqueeze(2)
            lengths = torch.LongTensor(lengths)            # print(captions.size())


            # Forward, Backward and Optimize
            img_dec_optim.zero_grad()
            img_enc_optim.zero_grad()

            txt_dec_optim.zero_grad()
            txt_enc_optim.zero_grad()

            # Image Auto_Encoder Forward

            encoder_outputs, Iz  = encoder_Img(images)
            IzI = decoder_Img(encoder_outputs)

            img_rc_loss = img_criterion(IzI,images)


            # Text Auto Encoder Forward

            # if cuda:
            #     src_seqs = src_seqs.cuda()
            # target = target[:-1] # exclude last target from inputs
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

            txt_rc_loss = 0

            for x,y in zip(TzT.transpose(0,1),glove_emb(captions).transpose(0,1)):
                if args.criterion == 'L1':
                    txt_rc_loss += txt_criterion(x,y)
                else:
                    txt_rc_loss = txt_criterion(x, y, Variable(torch.ones(x.size(0))).cuda())

            txt_rc_loss /= TzT.size(0)


            # Iz.requires_grad = False
            Tz = Tz[0].detach()


            if args.criterion == 'L1':
                cm_loss = cm_criterion(Iz, Tz)
            else:
                cm_loss = cm_criterion(Iz, Tz, Variable(torch.ones(args.batch_size)).cuda())


            # rate = 0.5
            # img_loss = img_rc_loss * (1 - rate) + cm_loss * rate
            # txt_loss = txt_rc_loss * (1 - rate) + cm_loss * rate
            img_loss = img_rc_loss + cm_loss
            txt_loss = txt_rc_loss + cm_loss


            img_losses.update(img_rc_loss.data[0],args.batch_size)
            txt_losses.update(txt_rc_loss.data[0],args.batch_size)
            cm_losses.update(cm_loss.data[0], args.batch_size)

            # Half of the times we update one pipeline the others the other one
            if not i % 2:
                img_loss.backward()
                img_enc_optim.step()
                img_dec_optim.step()
            else:
                txt_loss.backward()
                txt_enc_optim.step()
                txt_dec_optim.step()
            #
            # if (i+1) % args.log_interval == 0:
            #     print("---------------------")
            #     print("Img Loss: " + img_loss.avg)
            #     print("Txt Loss: " + txt_loss.avg)
            #     print("Cross-Modal Loss: " + cm_loss.avg)

            # Save the models
            if (i+1) % args.save_step == 0:
                torch.save(decoder_Img.state_dict(),
                           os.path.join(args.model_path,
                                        'decoder-img-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(encoder_Img.state_dict(),
                           os.path.join(args.model_path,
                                        'encoder-img-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(decoder_Txt.state_dict(),
                           os.path.join(args.model_path,
                                        'decoder-txt-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(encoder_Txt.state_dict(),
                           os.path.join(args.model_path,
                                        'encoder-txt-%d-%d.pkl' %(epoch+1, i+1)))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_Img: {img_l:.3f}| Loss_Txt: {txt_l:.3f} | Loss_CM: {cm_l:.3f}'.format(
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

        # VALIDATION TIME
        print('EPOCH ::: VALIDATION ::: ' + str(epoch + 1))

        # Set Evaluation Mode
        encoder_Img.eval()

        encoder_Txt.encoder.eval()

        # get pairs
        end = time.time()

        bar = Bar('Computing Validation Set Embeddings', max=len(val_loader))

        for i, (images, captions, lengths) in enumerate(val_loader):

            # Set mini-batch dataset
            images = to_var(images)
            captions = to_var(captions)

            captions = captions.transpose(0,1).unsqueeze(2)
            lengths = torch.LongTensor(lengths)

            _, img_emb = encoder_Img(images)

            txt_emb, _ = encoder_Txt(captions, lengths)


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
                        size=len(data_loader),
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        )
            bar.next()
        bar.finish()

        end = time.time()

        print("Computing Nearest Neighbors")
        neigh = NearestNeighbors(5)
        neigh.fit(result_embeddings[1])
        kneigh = neigh.kneighbors(result_embeddings[0], return_distance=False)

        bar = Bar('Computing top-K Accuracy', max=len(val_loader))

        top5 = AverageMeter()
        for i in range(result_embeddings.shape[1]):
            i_emb = result_embeddings[1][i]
            if i_emb in kneigh[i]:
                top5.update(1.)
            else:
                top5.update(0.)

            if not (i+1) % args.batch_size:
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Top-5 Acc {tp5:.3f}'.format(
                            batch= (i+1)/args.batch_size,
                            size=len(result_embeddings)/args.batch_size,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            tp5=top5.avg
                            )
                bar.next()
        bar.finish()










if __name__ == "__main__":
    main()
