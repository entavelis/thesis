import os
import torch.optim as optim
from models.CoupledVAE import *
from models.ImgDiscriminator import *
from models.SeqDiscriminator import *

from losses import *
from trainers.trainer import trainer
from utils import *
from itertools import chain

class coupled_vae_trainer(trainer):

    def __init__(self, args, embedding, vocab):
        super(coupled_vae_trainer,self).__init__(args, embedding, vocab)

        # Setting up the networks
        self.networks["coupled_vae"] = CoupledVAE(embedding,
                                                  len(vocab),
                                                  hidden_size=args.hidden_size,
                                                  latent_size=args.latent_size,
                                                  batch_size = args.batch_size,
                                                  img_dimension= args.crop_size)

        self.networks["img_discriminator"] = ImgDiscriminator(args.crop_size)

        self.networks["txt_discriminator"] = SeqDiscriminator(embedding, len(vocab))

        # Setting up the optimizers
        self.optimizers["coupled_vae"] = optim.Adam(valid_params(self.networks["coupled_vae"].parameters()),\
                                                    lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)

        self.optimizers["discriminators"] = optim.Adam(chain(self.networks["img_discriminator"].parameters(),\
                                                             self.networks["txt_discriminator"].parameters(),),\
                                                    lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)

        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1
        if args.cuda:
            self.one = self.one.cuda()
            self.mone = self.mone.cuda()

        self.recon_criterion = nn.MSELoss()

        self.nets_to_cuda()

        self.train_swapped = True # It's a GAN!!
        self.step = 0

        if args.load_vae != "NONE":
            self.load_vae(args.load_vae)
    def forward(self, epoch, images, captions, lengths, save_images):

        train_gen = self.iteration % 5
        # Forward, Backward and Optimize
        self.networks_zero_grad()

        # The generators parts
        img_gen, img_mu, img_logv, img_z, txt_gen, txt_mu, txt_logv, txt_z = \
                                 self.networks["coupled_vae"](images, captions, lengths, self.train_swapped)

        capt_emb = self.embeddings(captions)
        _, top_ind  = torch.topk(txt_gen,1)
        top_txt_gen = self.embeddings(top_ind[:,:,0])

        # Generator Loss
        img_rc_mu, img_rc_logv, img_rc_z, txt_rc_mu, txt_rc_logv, txt_rc_z = \
                                 self.networks["coupled_vae"].reconstruct(img_gen, top_txt_gen)

        # KL divergence for VAEs
        img_kl_loss = kl_loss(img_mu, img_logv).mean() + kl_loss(img_rc_mu, img_rc_logv).mean()
        txt_kl_loss = kl_loss(txt_mu, txt_logv).mean() + kl_loss(txt_rc_mu, txt_rc_logv).mean()

        # Intra-Modal
        im_emb_rc_loss = mse_loss(img_rc_z, img_z).mean()
        txt_emb_rc_loss = mse_loss(txt_rc_z, txt_z).mean()

        #Backwards

        # Discriminator Loss
        real_img = self.networks["img_discriminator"](images).mean()
        fake_img = self.networks["img_discriminator"](img_gen).mean()
        real_txt = self.networks["txt_discriminator"](capt_emb).mean()
        fake_txt = self.networks["txt_discriminator"](top_txt_gen).mean()

        # Gradient Penalty
        img_gp = calc_gradient_penalty(self.networks["img_discriminator"], images, img_gen)
        # txt_gp = calc_gradient_penalty(self.networks["txt_discriminator"], capt_emb, top_txt_gen)

        vae_loss = img_kl_loss + txt_kl_loss + im_emb_rc_loss + txt_emb_rc_loss + fake_img + fake_txt
        dis_loss = real_img - fake_img + real_txt - fake_txt + img_gp


        if train_gen:
            # Train VAE-Generators

            # To avoid computation as in wgan-gp
            # for p in self.networks["img_discriminator"].parameters():
            #     p.requires_grad = False  # to avoid computation

            # for p in self.networks["txt_discriminator"].parameters():
            #     p.requires_grad = False  # to avoid computation

            fake_img.backward(self.mone, retain_graph = True)
            fake_txt.backward(self.mone, retain_graph = True)
            vae_loss.backward()
            self.optimizers["coupled_vae"].step()

        else:

            # Train Discriminators
            # dis_loss.backward()

            real_img.backward(self.mone, retain_graph = True)
            fake_img.backward(self.one, retain_graph = True)
            real_txt.backward(self.mone, retain_graph = True)
            fake_txt.backward(self.one, retain_graph = True)

            # WGAN-GP
            img_gp.backward()
            # txt_gp.backward()

            self.optimizers["discriminators"].step()


        if save_images:
            self.save_samples(images[0], img_gen[0], captions[0], txt_gen[0])


        self.iteration += 1
        return vae_loss, dis_loss

    def backpropagate(self, loss):
        loss.backward()
        for opt in self.optimizers.values():
            opt.step()
        self.step += 1

    def load_vae(self, path2vae):
        try:
            self.networks["coupled_vae"].load_state_dict(torch.load(path2vae))

        except FileNotFoundError:
            print("Didn't find any models switching to training")

