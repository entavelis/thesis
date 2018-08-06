#Source: https://github.com/timbmg/Sentence-VAE/blob/master/model.py
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var
from losses import masked_cross_entropy
import torch.nn.functional as F
from torch.autograd import Variable
from losses import kl_anneal_function

from skipthoughts.skipthoughts import UniSkip

import sys

class CoupledVAE(nn.Module):
    def __init__(self,  embedding, vocab,
                img_dimension=256,
                hidden_size = 1024,
                latent_size = 512,
                num_layers=1,
                bidirectional = False,
                rnn_type = "GRU",
                word_dropout = 0.5,
                sos_idx = 1,
                eos_idx = 2,
                pad_idx = 0,
                max_sequence_length = 30,
                batch_size = 128,
                mask = 0.3,
                drop_out = 0.3,
                averaged_output = False,
                weight_sharing = True,
                use_variational = False):


        super(CoupledVAE, self).__init__()

        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        # like rnn-wgan
        self.averaged_output = True

        self.use_variational = use_variational
        if not use_variational:
            self.hidden_size = 1024
        else:
            self.hidden_size = hidden_size

        # self.uniskip = UniSkip("data/skip-thoughts", list(vocab.idx2word.items()), fixed_emb=False)
        vocab_size = len(vocab)
        self.vocab_size = vocab_size

        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        self.mask = mask
        self.common_z_size = int(mask * latent_size)
        self.left_z_size = latent_size - self.common_z_size

        self.latent_size = latent_size
        self.img_dimension = img_dimension


        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, img_dimension, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_dimension, img_dimension * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_dimension * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_dimension * 2, img_dimension * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_dimension * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_dimension * 4, img_dimension * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_dimension * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_dimension * 8, self.hidden_size, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_size, img_dimension * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(img_dimension * 8),
            nn.ReLU(True),
            # nn.Dropout2d(0.3,inplace=True),
            nn.ConvTranspose2d(img_dimension * 8, img_dimension * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_dimension * 4),
            nn.ReLU(True),
            # nn.Dropout2d(0.3,inplace=True),
            nn.ConvTranspose2d(img_dimension * 4, img_dimension * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_dimension * 2),
            nn.ReLU(True),
            # nn.Dropout2d(0.3,inplace=True),
            nn.ConvTranspose2d(img_dimension * 2,     img_dimension, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_dimension),
            nn.ReLU(True),
            # nn.Dropout2d(0.3,inplace=True),
            nn.ConvTranspose2d(img_dimension,      3, 4, 2, 1, bias=False),
            nn.Tanh()
            # nn.Sigmoid()
        )

        # self.batch_norm_txt = nn.BatchNorm1d(self.hidden_size)
        self.hidden2mean_img = nn.Linear(self.hidden_size, latent_size)
        self.hidden2logv_img = nn.Linear(self.hidden_size, latent_size)
        self.latent2hidden_img = nn.Linear(latent_size, self.hidden_size)

        if use_variational:
            self.hidden2mean_txt = nn.Linear(1024, latent_size)
            self.hidden2logv_txt = nn.Linear(1024, latent_size)
            self.latent2hidden_txt = nn.Linear(latent_size, self.hidden_size)

        self.noise = torch.FloatTensor(self.batch_size, 100).cuda()


    def forward(self, input_images, input_emb, ln):
        txt_enc = input_emb #self.uniskip(input_captions, list(lengths.data))
        # Image pathway
        img_enc = self.img_encoder_forward(input_images)
        # img_enc = self.batch_norm_img(img_enc)
        img_mu, img_logv, img_z = self.Hidden2Z_img(img_enc)
        hidden4img2img = self.Z2Hidden_img4img(img_z)
        img2img_out = self.img_decoder_forward(hidden4img2img)

        # noise = to_var(self.noise.resize_(self.batch_size, 100).normal_(0,1))
        # txt2img_out = self.img_decoder_forward(torch.cat([txt_enc, noise],-1))

        # Generation txt2img
        if self.use_variational:
           txt_mu, txt_logv, txt_z = self.Hidden2Z_txt(txt_enc)
           hidden4txt2img = self.Z2Hidden_txt4img(txt_z)
           txt2img_out = self.img_decoder_forward(hidden4txt2img)
        else:
           txt_z = txt_enc
           noise = to_var(self.noise.resize_(self.batch_size, 100).normal_(0,1))
           txt2img_out = self.img_decoder_forward(torch.cat([txt_enc, noise],-1))

        # img_noise = to_var(self.img_noise.resize_(self.batch_size, self.left_z_size).normal_(0,1))



        if self.use_variational:
            return img2img_out, txt2img_out,  img_z, txt_enc, txt_z, img_mu, img_logv,  txt_mu, txt_logv
        else:
            return img2img_out, txt2img_out, img_enc, txt_enc, img_mu, img_logv

    def image_reconstruction_loss(self, original, reconstructed, mu=0, logvar=1, beta = 3):
                # print(recon_x.size(), x.size())
        flat_dim = original.size(2)**2
        flat_rc_x = reconstructed.view(-1, flat_dim)
        flat_x = original.view(-1, flat_dim)

        # BCE = F.binary_cross_entropy(flat_rc_x, flat_x, size_average=False)
        RC = torch.sum(torch.abs(flat_rc_x - flat_x)**2)
        # L2 loss too blurry
        # RC = F.mse_loss(flat_rc_x, flat_x, size_average=False)

        if self.use_variational:
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # return BCE + KLD
            return (RC + beta * KLD)/ (self.batch_size * (self.img_dimension ** 2))
        else:
            return RC/ (self.batch_size * (self.img_dimension ** 2))

        return mu, logv, z

    def Hidden2Z_img(self, hidden):
        mu = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)

        # REPARAMETERIZATION
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([self.batch_size, self.latent_size]))
        z = z * std + mu

        return mu, logv, z

    def Z2Hidden_img4img(self, z):
        return self.latent2hidden(z)

    def Hidden2Z_txt(self, hidden):
        mu = self.hidden2mean_txt(hidden)
        logv = self.hidden2logv_txt(hidden)

        # REPARAMETERIZATION
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([self.batch_size, self.latent_size]))
        z = z * std + mu

        return mu, logv, z

    def Hidden2Z_img(self, hidden):
        mu = self.hidden2mean_img(hidden)
        logv = self.hidden2logv_img(hidden)

        # REPARAMETERIZATION
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([self.batch_size, self.latent_size]))
        z = z * std + mu

        return mu, logv, z

    def Z2Hidden_img4img(self, z):
        return self.latent2hidden_img(z)

    def Z2Hidden_txt4img(self, z):
        return self.latent2hidden_txt(z)

    def img_encoder_forward(self, input):
        enc = self.encoder_cnn(input)
        enc = enc.view(self.batch_size,-1)
        return enc

    def img_decoder_forward(self, input):
        output = self.decoder_cnn(input.view(self.batch_size, self.hidden_size, 1 , 1))
        return  output

    def reconstruct(self, gen_images):

        img_enc = self.img_encoder_forward(gen_images)

        img_mu, img_logv, img_z = self.Hidden2Z_img(img_enc)

        return img_mu, img_logv, img_z
