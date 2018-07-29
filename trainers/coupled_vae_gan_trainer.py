import os
import torch.optim as optim
from models.CoupledVAE import *
from models.ImgDiscriminator import *
from models.SeqDiscriminator import *

from losses import *
from trainers.trainer import trainer
from utils import *
from itertools import chain

from vae_validate import validate as val

class coupled_vae_gan_trainer(trainer):

    def __init__(self, args, embedding, vocab):
        super(coupled_vae_gan_trainer,self).__init__(args, embedding, vocab)

        self.use_variational = args.variational == "true"
        weight_sharing = args.weight_sharing == "true"
        # Setting up the networks
        self.networks["coupled_vae"] = CoupledVAE(embedding,
                                                  len(vocab),
                                                  hidden_size=args.hidden_size,
                                                  latent_size=args.latent_size,
                                                  batch_size = args.batch_size,
                                                  img_dimension= args.crop_size,
                                                  mask= args.common_emb_ratio,
                                                  use_variational= self.use_variational,
                                                  weight_sharing= weight_sharing)

        self.networks["img_discriminator"] = ImgDiscriminator(args.crop_size, batch_size = args.batch_size,
                                                              mask = args.common_emb_ratio,
                                                              latent_size= args.latent_size)

        self.networks["txt_discriminator"] = SeqDiscriminator(embedding, len(vocab),
                                                              mask =args.common_emb_ratio,
                                                              latent_size= args.latent_size)

        # Setting up the optimizers
        self.optimizers["coupled_vae"] = optim.Adam(valid_params(self.networks["coupled_vae"].parameters()),\
                                                    lr=args.learning_rate, betas=(0.5, 0.999)) #, weight_decay=0.00001)

        self.optimizers["discriminators"] = optim.Adam(chain(self.networks["img_discriminator"].parameters(),\
                                                    valid_params(self.networks["txt_discriminator"].parameters()),),\
                                                    lr=args.learning_rate, betas=(0.5, 0.999)) #, weight_decay=0.00001)

        # Setting up the losses
        self.create_losses_meter(["Ls_D_img", "Ls_D_txt",
                                  "Ls_G_img",  "Ls_G_txt",
                                  "Ls_D_img_rl", "Ls_D_txt_rl",
                                  "Ls_D_img_fk","Ls_D_txt_fk",
                                  "Ls_GP_img", "Ls_GP_txt",
                                  "Ls_RC_img", "Ls_RC_txt", "Ls_RC_txt_fk",
                                  # "Ls_KL_img", "Ls_KL_txt",
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
        self.iteration += 1

        if not self.iteration % self.args.model_save_interval and not self.keep_loading:
            self.save_models(self.iteration)

        # Save samples
        if save_images:
            if self.use_variational:
                img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_z, txt_z, img_mu, img_logv, txt_mu, txt_logv = \
                                 self.networks["coupled_vae"](images, captions , lengths)
            else:
                img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_z, txt_z= \
                                 self.networks["coupled_vae"](images, captions , lengths)

            self.save_samples(img2img_out[0], txt2img_out[0], txt2txt_out[0], img2txt_out[0])

        if self.iteration < self.args.vae_pretrain_iterations:
            self.train_G(epoch,images,captions,lengths, pretrain= True)
            return

        if self.iteration < self.args.vae_pretrain_iterations + self.args.disc_pretrain_iterations:
            cycle = 101
        else:
            cycle = 6

        # cycle = 60

        # train_gen = self.iteration > 500 and self.iteration % 6
        if not self.iteration % cycle:
        # if not self.iteration % cycle or max(self.losses["Ls_G_img"].val,self.losses["Ls_G_txt"].val) > 0.75:
        # if self.iteration % cycle < 50:
             #
            for p in self.networks["img_discriminator"].parameters():  # reset requires_grad
                p.requires_grad = False  # they are set to False below in netG update
            for p in self.networks["txt_discriminator"].parameters():  # reset requires_grad
                p.requires_grad = False  # they are set to False below in netG update

            self.train_G(epoch, images, captions, lengths)
            # self.optimizers["coupled_vae"].step()
            #
            for p in self.networks["img_discriminator"].parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            for p in self.networks["txt_discriminator"].parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
        else:

            self.train_D(epoch, images, captions, lengths)

        # Log Losses:
        # self.losses["GEN_loss"].update(self.gen_loss.data[0],self.args.batch_size)
        # self.losses["DIS_loss"].update(self.dis_loss.data[0],self.args.batch_size)



    def train_D(self,epoch, images, captions, lengths):
        # train with real
        # self.networks["img_discriminator"].zero_grad()
        # self.networks["txt_discriminator"].zero_grad()
        self.networks_zero_grad()
        # clamp parameters to a cube
        for p in self.networks["img_discriminator"].parameters():
            p.data.clamp_(-0.01, 0.01)

        # clamp parameters to a cube
        for p in self.networks["txt_discriminator"].parameters():
            p.data.clamp_(-0.01, 0.01)



        images = add_gaussian(images, std=0.05 * (self.args.epoch_size - epoch)/self.args.epoch_size)
        # train with fake
        if self.use_variational:
            img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_z, txt_z, img_mu, img_logv, txt_mu, txt_logv, = \
                                 self.networks["coupled_vae"](images.detach(),captions.detach(), lengths)
        else:
            img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_z, txt_z= \
                                 self.networks["coupled_vae"](images.detach(),captions.detach(), lengths)

        errD_img_real = self.networks["img_discriminator"](images.detach(), txt_z.detach())
        errD_img_real.backward(self.one, retain_graph = True)

        errD_txt_real, _ = self.networks["txt_discriminator"](captions.detach(), img_z.detach(), lengths)
        errD_txt_real.backward(self.one, retain_graph = True)
        errD_img_fake = self.networks["img_discriminator"](txt2img_out.detach(), txt_z.detach())
        # errD_img_fake.backward(self.mone, retain_graph = True)
        errD_img_fake.backward(self.mone)

        gen_lengths = get_gen_lengths(img2txt_out)
        errD_txt_fake, _ = self.networks["txt_discriminator"](img2txt_out.detach(), img_z.detach(), gen_lengths)
        # errD_txt_fake.backward(self.mone, retain_graph = True)
        errD_txt_fake.backward(self.mone)

        # errGP_img = calc_gradient_penalty(self.networks["img_discriminator"], images, txt2img_out, img=True)
        # errGP_img.backward()

        # errGP_txt = calc_gradient_penalty(self.networks["txt_discriminator"], captions, img2txt_out, img=False)
        # errGP_txt.backward()

        errD_img = errD_img_real - errD_img_fake # + errGP_img
        errD_txt = errD_txt_real - errD_txt_fake # + errGP_txt
        self.optimizers["discriminators"].step()

        self.losses["Ls_D_img"].update(errD_img.data[0], self.args.batch_size)
        self.losses["Ls_D_txt"].update(errD_txt.data[0], self.args.batch_size)
        self.losses["Ls_D_img_fk"].update(errD_img_fake.data[0], self.args.batch_size)
        self.losses["Ls_D_txt_fk"].update(errD_txt_fake.data[0], self.args.batch_size)
        self.losses["Ls_D_img_rl"].update(errD_img_real.data[0], self.args.batch_size)
        self.losses["Ls_D_txt_rl"].update(errD_txt_real.data[0], self.args.batch_size)
        # self.losses["Ls_GP_img"].update(errGP_img.data[0], self.args.batch_size)
        # self.losses["Ls_GP_txt"].update(errGP_txt.data[0], self.args.batch_size)

        # Gradient Penalty

       # img_gp = calc_gradient_penalty(self.networks["discriminator"], images, img_gen)


    def train_G(self, epoch, images, captions, lengths, pretrain = False):

        # Forward, Backward and Optimize
        self.networks_zero_grad()

        if self.use_variational:
            img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_z, txt_z, img_mu, img_logv, txt_mu, txt_logv = \
                                 self.networks["coupled_vae"](images, captions, lengths)

            img_rc_loss = img_vae_loss(img2img_out, images, img_mu, img_logv) /\
                           (self.args.batch_size * (self.args.crop_size**2))

            NLL_loss, KL_loss, KL_weight = seq_vae_loss(txt2txt_out, captions,
                                                   lengths, txt_mu, txt_logv, "logistic", self.step, 0.0025,
                                                   2500)
            txt_rc_loss = (NLL_loss + KL_weight * KL_loss) / torch.sum(lengths).float()
        else:

            img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_z, txt_z =\
                self.networks["coupled_vae"](images, captions, lengths)

            img_rc_loss = self.networks["coupled_vae"].image_reconstruction_loss(images, img2img_out)
            txt_rc_loss = self.networks["coupled_vae"].text_reconstruction_loss(captions, txt2txt_out, lengths)

        vae_loss = txt_rc_loss + img_rc_loss
        self.step += 1

        # Only retain graph if not on pretraining mode
        vae_loss.backward(retain_graph = not pretrain)

        if not pretrain:
            lambda_G = 1

            gen_len = get_gen_lengths(img2txt_out)
            errG_txt, _ = self.networks["txt_discriminator"](img2txt_out, img_z, gen_len) * lambda_G
            errG_txt.backward(self.one)

            errG_img = self.networks["img_discriminator"](txt2img_out, txt_z) * lambda_G
            errG_img.backward(self.one)

        self.optimizers["coupled_vae"].step()

        if not pretrain:
            self.losses["Ls_G_img"].update(errG_img.data[0], self.args.batch_size)
            self.losses["Ls_G_txt"].update(errG_txt.data[0], self.args.batch_size)

        self.losses["Ls_RC_img"].update(img_rc_loss.data[0], self.args.batch_size)
        self.losses["Ls_RC_txt"].update(txt_rc_loss.data[0], self.args.batch_size)
        self.losses["Ls_VAE"].update(vae_loss.data[0], self.args.batch_size)

    def validate(self, loader):
        val(self.networks["coupled_vae"], loader, int(self.args.common_emb_ratio * self.args.latent_size),
            limit = 10, metric='euclidean')

    def evaluate(self, epoch, images, captions, lengths):

        if self.use_variational:
            img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_z, txt_z, img_mu, img_logv, txt_mu, txt_logv = \
                                 self.networks["coupled_vae"](images, captions, lengths)

            img_rc_loss = img_vae_loss(img2img_out, images, img_mu, img_logv) /\
                           (self.args.batch_size * (self.args.crop_size**2))

            NLL_loss, KL_loss, KL_weight = seq_vae_loss(txt2txt_out, captions,
                                                   lengths, txt_mu, txt_logv, "logistic", self.step, 0.0025,
                                                   2500)
            txt_rc_loss = (NLL_loss + KL_weight * KL_loss) / torch.sum(lengths).float()
        else:

            img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_z, txt_z =\
                self.networks["coupled_vae"](images, captions, lengths)

            img_rc_loss = self.networks["coupled_vae"].image_reconstruction_loss(images, img2img_out)
            txt_rc_loss = self.networks["coupled_vae"].text_reconstruction_loss(captions, txt2txt_out, lengths)
            txt_rc_loss = self.networks["coupled_vae"].text_reconstruction_loss(captions, txt2txt_out, lengths)

        vae_loss = txt_rc_loss + img_rc_loss




