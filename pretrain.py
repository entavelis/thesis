import os
import argparse
from itertools import chain

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

from image_caption.build_vocab import Vocabulary
from image_caption.data_loader import get_loader



from progressbar import ETA, Bar, Percentage, ProgressBar

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

parser.add_argument('--crop_size', type=int, default=224,
                    help='size for randomly cropping images')
parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                    help='path for vocabulary wrapper')
parser.add_argument('--image_dir', type=str, default='./data/resized2014',
                    help='directory for resized images')
parser.add_argument('--embedding_size', type=int, default=100)

parser.add_argument('--embedding_path', type=str,
                    default='./glove/',
                    help='path for pretrained embeddings')
parser.add_argument('--caption_path', type=str,
                    default='./data/annotations/captions_train2014.json',
                    help='path for train annotation json file')
parser.add_argument('--log_step', type=int, default=10,
                    help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=1000,
                    help='step size for saving trained models')

# Model parameters
parser.add_argument('--embed_size', type=int, default=256,
                    help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int, default=512,
                    help='dimension of lstm hidden states')
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in lstm')

parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.001)


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

def main():
    # global args
    args = parser.parse_args()

    cuda = args.cuda
    if cuda == 'true':
        cuda = True
    else:
        cuda = False

    # Image preprocessing
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
    emb = nn.Embedding(emb.size(0), emb.size(1))
    emb.weight = nn.Parameter(emb)

    # Build data loader
    print("Building Data Loader...")
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    print("Setting up the Networks...")
    #     generator_A = Generator()
    encoder_Img = TextEncoder(emb)
    decoder_Txt = ImageDecoder(TextEncoder, emb)

    #     generator_B = Generator()
    encoder_Txt = ImageEncoder()
    decoder_Img = TextDecoder()

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
    txt_criterion = nn.CrossEntropyLoss()
    cm_criterion = nn.MSELoss()

    #     gen_params = chain(generator_A.parameters(), generator_B.parameters())
    print("Setting up the Optimizers...")
    img_params = chain(decoder_Img.parameters(), encoder_Img.parameters())
    txt_params = chain(decoder_Txt.parameters(), encoder_Txt.parameters())

    # ATTENTION: Check betas and weight decay
    img_optim = optim.Adam( img_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)
    txt_optim = optim.Adam( txt_params, lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)


    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        #         We don't want our data to be shuffled
        #         data_style_A, data_style_B = shuffle_data( data_style_A, data_style_B)

        widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
        pbar = ProgressBar(maxval=total_step, widgets=widgets)
        pbar.start()

        for i, (images, captions, lengths) in enumerate(data_loader):
            pbar.update(i)

            print(captions)

            # Set mini-batch dataset
            images = to_var(images, volatile=True)
            captions = to_var(captions)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            #Forward, Backward and Optimize
            encoder_Img.zero_grad()
            decoder_Img.zero_grad()

            encoder_Txt.zero_grad()
            decoder_Txt.zero_grad()

            Iz = encoder_Img(images)
            IzI = decoder_Img(Iz)

            img_rc_loss = img_criterion(IzI,images)

            for cap in captions:
                Tz = encoder_Txt(captions)
                TzT = decoder_Txt(Tz)
                txt_rc_loss = txt_criterion(TzT,captions)
                cm_loss = cm_criterion(Iz,Tz)


            rate = 0.9
            img_loss = img_rc_loss * (1 - rate) + cm_loss * rate
            txt_loss = txt_rc_loss * (1 - rate) + cm_loss * rate

            # Half of the times we update one pipeline the others the other one
            if i % 2 == 0:
                img_loss.backward()
                img_optim.step()
            else:
                txt_loss.backward()
                txt_optim.step()

            if i % args.log_interval == 0:
                print("---------------------")
                print("Img Loss: " + as_np(img_rc_loss.mean()))
                print("Txt Loss: " + as_np(img_rc_loss.mean()))
                print("Cross-Modal Loss: " + as_np(cm_loss.mean()))

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


if __name__ == "__main__":
    main()
