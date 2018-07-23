import os
import torch.optim as optim
from models.CoupledVAE import *
from models.ImgDiscriminator import *
from models.SeqDiscriminator import *

from losses import *
from trainers.trainer import trainer
from utils import *
from itertools import chain

class coupled_vae_gan_trainer(trainer):

    def __init__(self, args, embedding, vocab):
        super(coupled_vae_gan_trainer,self).__init__(args, embedding, vocab)

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
                                                    valid_params(self.networks["txt_discriminator"].parameters()),),\
                                                    lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)

        # Setting up the losses
        self.create_losses_meter(["Ls_D_img", "Ls_D_img",
                                  "Ls_G_img",  "Ls_G_img",
                                  "Ls_D_img_rl", "Ls_D_img_rl",
                                  "Ls_D_img_fk","Ls_D_img_fk",
                                  "Ls_GP_img", "Ls_GP_img",
                                  "Ls_VAE"])

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

        if self.iteration < 2500:
            cycle = 101
        else:
            cycle = 6

        # train_gen = self.iteration > 500 and self.iteration % 6
        if not self.iteration % cycle:
             #
            for p in self.networks["img_discriminator"].parameters():  # reset requires_grad
                p.requires_grad = False  # they are set to False below in netG update
            for p in self.networks["txt_discriminator"].parameters():  # reset requires_grad
                p.requires_grad = False  # they are set to False below in netG update

            self.train_G(epoch, images, captions, lengths)
            self.optimizers["generator"].step()
            #
            for p in self.networks["img_discriminator"].parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            for p in self.networks["txt_discriminator"].parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
        else:

            self.train_D(epoch, images, captions, lengths)

        self.iteration += 1
        # Log Losses:
        # self.losses["GEN_loss"].update(self.gen_loss.data[0],self.args.batch_size)
        # self.losses["DIS_loss"].update(self.dis_loss.data[0],self.args.batch_size)

        # if save_images:
        #     self.save_samples(images[0], self.networks["generator"](self.fixed_noise)[0], captions[0], captions[0])



    def train_D(self,epoch, images, captions, lengths):

        # clamp parameters to a cube
        for p in self.networks["img_discriminator"].parameters():
            p.data.clamp_(-0.01, 0.01)

        # clamp parameters to a cube
        for p in self.networks["txt_discriminator"].parameters():
            p.data.clamp_(-0.01, 0.01)

        # train with real
        self.networks["img_discriminator"].zero_grad()
        self.networks["txt_discriminator"].zero_grad()

        errD_img_real = self.networks["img_discriminator"](images)
        errD_img_real.backward(self.one)

        errD_txt_real = self.networks["txt_discriminator"](captions)
        errD_txt_real.backward(self.one)

        # train with fake
        img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_mu, img_logv, img_z, txt_mu, txt_logv, txt_z = \
                                 self.networks["coupled_vae"](images.detach(),captions.detach(), lengths)

        errD_img_fake = self.networks["img_discriminator"](txt2img_out)
        errD_img_fake.backward(self.mone)

        errD_txt_fake = self.networks["txt_discriminator"](img2txt_out)
        errD_txt_fake.backward(self.mone)

        errGP_img = calc_gradient_penalty(self.networks["img_discriminator"], images, txt2img_out)
        errGP_img.backward()

        errGP_txt = calc_gradient_penalty(self.networks["txt_discriminator"], captions, img2txt_out)
        errGP_txt.backward()

        errD_img = errD_img_real - errD_img_fake + errGP_img
        errD_txt = errD_txt_real - errD_txt_fake + errGP_txt
        self.optimizers["discriminator"].step()

        self.losses["Ls_D_img"].update(errD_img.data[0], self.args.batch_size)
        self.losses["Ls_D_txt"].update(errD_txt.data[0], self.args.batch_size)
        self.losses["Ls_D_img_fk"].update(errD_img_fake.data[0], self.args.batch_size)
        self.losses["Ls_D_txt_fk"].update(errD_txt_fake.data[0], self.args.batch_size)
        self.losses["Ls_D_img_rl"].update(errD_img_real.data[0], self.args.batch_size)
        self.losses["Ls_D_txt_rl"].update(errD_txt_real.data[0], self.args.batch_size)
        self.losses["Ls_GP_img"].update(errGP_img.data[0], self.args.batch_size)
        self.losses["Ls_GP_txt"].update(errGP_txt.data[0], self.args.batch_size)

        # Gradient Penalty

       # img_gp = calc_gradient_penalty(self.networks["discriminator"], images, img_gen)


    def train_G(self,epoch, images, captions, lengths):
        self.networks["coupled_vae"].zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_mu, img_logv, img_z, txt_mu, txt_logv, txt_z = \
            self.networks["coupled_vae"](images, captions, lengths)

        errG_img = self.networks["img_discriminator"](txt2img_out)
        errG_img.backward(self.one)

        errG_txt = self.networks["txt_discriminator"](img2txt_out)
        errG_txt.backward(self.one)

         # KL divergence for VAEs
        lambda_KL = 0.1
        errKL_img = kl_loss(img_mu, img_logv).mean(0)
        errKL_txt = kl_loss(txt_mu, txt_logv).mean(0)

        # Reconstruction error
        lambda_RC = 100
        errRC_img = recon_loss_img(img2img_out, target=images)
        errRC_txt = recon_loss_txt(txt2txt_out, target=captions)

        vae_loss = lambda_KL * (errKL_img + errKL_txt) + lambda_RC * (errRC_img + errRC_txt)
        vae_loss.backward(retain_graph = True)



        self.optimizers["coupled_vae"].step()
        self.losses["Ls_G_img"].update(errG_img.data[0], self.args.batch_size)
        self.losses["Ls_G_txt"].update(errG_txt.data[0], self.args.batch_size)
        self.losses["Ls_VAE"].update(vae_loss.data[0], self.args.batch_size)

    def forward_old(self, epoch, images, captions, lengths, save_images):


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
        txt_gp = calc_gradient_penalty(self.networks["txt_discriminator"], capt_emb, top_txt_gen)

        lambda_KL = 0.1
        lambda_intra_modal = 100
        lambda_gen = 10
        vae_loss = lambda_KL * (img_kl_loss + txt_kl_loss) + \
                   lambda_intra_modal * (im_emb_rc_loss + txt_emb_rc_loss) +\
                   lambda_gen * (fake_img + fake_txt)
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
            img_gp.backward(retain_graph=True)
            txt_gp.backward(retain_graph=True)

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

