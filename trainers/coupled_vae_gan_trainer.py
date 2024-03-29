import os
import torch.optim as optim
from models.CoupledVAE import *
from models.ImgDiscriminator import *
from models.SeqDiscriminator import *

from losses import *
from metrics import *

from trainers.trainer import trainer
from utils import *
from itertools import chain

from vae_validate import validate as val

class coupled_vae_gan_trainer(trainer):

    def __init__(self, args, embedding, vocab):
        super(coupled_vae_gan_trainer,self).__init__(args, embedding, vocab)

        self.use_variational = args.variational == "true"
        self.freeze_encoders = args.freeze_encoders == "true"
        self.use_feature_matching = args.use_feature_matching == "true"
        self.use_gradient_penalty = args.use_gradient_penalty == "true"
        self.double_fake_error = args.use_double_fake_error == "true"
        self.use_gumbel_generator = args.use_gumbel_generator == "true"
        self.cycle_consistency = args.cycle_consistency == "true"
        self.use_parallel = args.use_parallel_recon == "true"

        if args.cycle_consistency_criterion == "mse":
            self.cc_criterion = nn.MSELoss(size_average=True)
        else:
            self.cc_criterion = nn.HingeEmbeddingLoss()


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
                                                  weight_sharing= weight_sharing,
                                                  use_gumbel_generator = self.use_gumbel_generator)

        self.networks["img_discriminator"] = ImgDiscriminator(args.crop_size, batch_size = args.batch_size,
                                                              mask = args.common_emb_ratio,
                                                              latent_size= args.latent_size)

        self.networks["txt_discriminator"] = SeqDiscriminator(embedding, len(vocab),
                                                              hidden_size=args.hidden_size,
                                                              mask =args.common_emb_ratio,
                                                              latent_size= args.latent_size,
                                                              use_gumbel_generator = self.use_gumbel_generator)

        # Initialiaze the weights
        self.networks["img_discriminator"].apply(weights_init)
        self.networks["coupled_vae"].apply(weights_init)
        # Setting up the optimizers
        self.optimizers["coupled_vae"] = optim.Adam(valid_params(self.networks["coupled_vae"].parameters()),\
                                                    lr=args.learning_rate, betas=(0.5, 0.999)) #, weight_decay=0.00001)

        self.optimizers["discriminators"] = optim.Adam(chain(self.networks["img_discriminator"].parameters(), \
                                                       valid_params(self.networks["txt_discriminator"].parameters()), ), \
                                                       lr=args.learning_rate, betas=(0.5, 0.999))  # , weight_decay=0.00001)
        self.masked_latent_size = int(args.common_emb_ratio * args.latent_size)

        # Setting up the losses
        losses = ["Ls_D_img", "Ls_D_txt",
                                  "Ls_G_img",  "Ls_G_txt",
                                  "Ls_D_img_rl", "Ls_D_txt_rl",
                                  "Ls_D_img_fk","Ls_D_txt_fk",
                                  "Ls_RC_img", "Ls_RC_txt", "Ls_RC_txt_fk",
                                  # "Ls_KL_img", "Ls_KL_txt",
                                  "Ls_VAE"]



        if self.use_gradient_penalty:
            losses += ["Ls_GP_img", "Ls_GP_txt"]

        if self.use_feature_matching:
            losses.append("Ls_FM_img")

        if self.double_fake_error:
            losses.append("Ls_D_img_wr")
            losses.append("Ls_D_txt_wr")

        if self.cycle_consistency:
            losses.append("Ls_CC_txt")
            losses.append("Ls_CC_img")

        self.create_losses_meter(losses)

        # Set Evaluation Metrics
        metrics = ["Ls_RC_txt",
                                  "Ls_D_img_rl", "Ls_D_txt_rl",
                                  "Ls_D_img_fk","Ls_D_txt_fk",
                                  "BLEU", "TxtL2"]

        if self.double_fake_error:
            metrics.append("Ls_D_img_wr")
            metrics.append("Ls_D_txt_wr")

        self.create_metrics_meter(metrics)
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

        self.epoch_max_len = Variable(torch.zeros(self.args.batch_size).long().cuda())

    def forward(self, epoch, images, captions, lengths, save_images):
        self.iteration += 1

        # if epoch >  self.args.mixed_training_epoch:
        #     train_img = True
        #     train_txt = True
        # elif not epoch % 2:
        #     train_img = True
        #     train_txt = False
        # else:
        #     train_img = False
        #     train_txt = True

        train_txt = True
        train_img = False
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

            self.save_samples(images[0], img2img_out[0], txt2img_out[0], captions[0], txt2txt_out[0], img2txt_out[0])

        if self.iteration < self.args.vae_pretrain_iterations:
            self.train_G(epoch,images,captions,lengths,
                         pretrain= True, train_img=train_img, train_txt=train_txt)
            return


        if self.freeze_encoders and self.iteration == self.args.vae_pretrain_iterations:
            self.networks["coupled_vae"].encoder_rnn.requires_grad = False
            self.networks["coupled_vae"].encoder_cnn.requires_grad = False
            if self.use_variational:
                self.networks["coupled_vae"].hidden2mean_txt.requires_grad = False
                self.networks["coupled_vae"].hidden2mean_img.requires_grad = False
                self.networks["coupled_vae"].hidden2logv_txt.requires_grad = False
                self.networks["coupled_vae"].hidden2logv_img.requires_grad = False

            self.optimizers["coupled_vae"] = optim.Adam(valid_params(self.networks["coupled_vae"].parameters()),\
                                                    lr=self.args.learning_rate, betas=(0.5, 0.999)) #, weight_decay=0.00001)


        # self.epoch_max_len.fill_(5+int(epoch/2))
        # lengths = torch.min(self.epoch_max_len, lengths)

        # if self.iteration < self.args.vae_pretrain_iterations + self.args.disc_pretrain_iterations:
        #     cycle = 101
        # else:
        #     cycle = self.args.gan_cycle

        cycle = self.args.gan_cycle

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

            self.train_G(epoch, images, captions, lengths, train_img = train_img,
                         train_txt= train_txt)
            # self.optimizers["coupled_vae"].step()
            #
            for p in self.networks["img_discriminator"].parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            for p in self.networks["txt_discriminator"].parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
        else:

            self.train_D(epoch, images, captions, lengths, train_img, train_txt)

        # Log Losses:
        # self.losses["GEN_loss"].update(self.gen_loss.data[0],self.args.batch_size)
        # self.losses["DIS_loss"].update(self.dis_loss.data[0],self.args.batch_size)



    def train_D(self,epoch, images, captions, lengths, train_img = True, train_txt=True):

        # train with real
        # self.networks["img_discriminator"].zero_grad()
        # self.networks["txt_discriminator"].zero_grad()
        self.networks_zero_grad()
        # clamp parameters to a cube
        if not self.use_gradient_penalty:
            for p in self.networks["img_discriminator"].parameters():
                p.data.clamp_(- self.args.img_clamping, self.args.img_clamping)

        # clamp parameters to a cube
        for p in self.networks["txt_discriminator"].parameters():
            p.data.clamp_(- self.args.txt_clamping, self.args.txt_clamping)



        # train with fake
        if self.use_variational:
            img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_z, txt_z, img_mu, img_logv, txt_mu, txt_logv, = \
                                 self.networks["coupled_vae"](images.detach(),captions.detach(), lengths)
        else:
            img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_z, txt_z= \
                                 self.networks["coupled_vae"](images.detach(),captions.detach(), lengths)

        if epoch < 200:
            images = add_gaussian(images, std=0.05 * (200 - epoch)/200)
            txt2img_out = add_gaussian(txt2img_out, std=0.05 * (200 - epoch)/200)

        perm_idz = torch.randperm(self.args.batch_size).cuda()


        if train_txt:
            errD_txt_real, _ = self.networks["txt_discriminator"](captions.detach(), img_z.detach(), lengths)
            errD_txt_real.backward(self.one, retain_graph = True)

            gen_lengths = get_gen_lengths(img2txt_out)

            errD_txt_fake = self.networks["txt_discriminator"](img2txt_out.detach(), img_z.detach(), gen_lengths)[0]
            if self.double_fake_error:
                # errD_txt_fake = (self.networks["txt_discriminator"](img2txt_out.detach(), img_z.detach(), gen_lengths)[0]+
                errD_txt_wrong = self.networks["txt_discriminator"](captions.detach(), img_z[perm_idz].detach(), lengths)[0]
                errD_txt_wrong /= 2
                errD_txt_wrong.backward(self.mone)
                errD_txt_fake /= 2
            # errD_txt_fake.backward(self.mone, retain_graph = True)
            errD_txt_fake.backward(self.mone)
            errD_txt = errD_txt_real - errD_txt_fake # + errGP_txt

        if train_img:
            errD_img_real, act_real = self.networks["img_discriminator"](images.detach(), txt_z.detach())
            errD_img_real.backward(self.one, retain_graph = True)

            errD_img_fake, act_fake = self.networks["img_discriminator"](txt2img_out.detach(), txt_z.detach())
            if self.double_fake_error:
                # errD_img_fake = (self.networks["img_discriminator"](txt2img_out.detach(), txt_z.detach())[0] +
                errD_img_wrong = self.networks["img_discriminator"](images.detach(), txt_z[perm_idz].detach())[0]
                errD_img_wrong /= 2
                errD_img_wrong.backward(self.mone)
                errD_img_fake /= 2

            errD_img_fake.backward(self.mone)
            # errD_img_fake.backward(self.mone)
            errD_img = errD_img_real - errD_img_fake  #+ errGP_img


            if self.use_gradient_penalty:
                errGP_img = calc_gradient_penalty(self.networks["img_discriminator"], images.detach(), txt2img_out.detach(),
                                                emb= txt_z.detach(), img=True)
                errGP_img.backward()
                errD_img += errGP_img


        # errGP_txt = calc_gradient_penalty(self.networks["txt_discriminator"], captions, img2txt_out, img=False)
        # errGP_txt.backward()


        self.optimizers["discriminators"].step()

        if train_img:
            self.losses["Ls_D_img"].update(errD_img.data[0], self.args.batch_size)
            self.losses["Ls_D_img_fk"].update(errD_img_fake.data[0], self.args.batch_size)
            self.losses["Ls_D_img_rl"].update(errD_img_real.data[0], self.args.batch_size)
            if self.use_gradient_penalty:
                self.losses["Ls_GP_img"].update(errGP_img.data[0], self.args.batch_size)
            if self.double_fake_error:
                self.losses["Ls_D_img_wr"].update(errD_img_wrong.data[0], self.args.batch_size)

        if train_txt:
            self.losses["Ls_D_txt"].update(errD_txt.data[0], self.args.batch_size)
            self.losses["Ls_D_txt_fk"].update(errD_txt_fake.data[0], self.args.batch_size)
            self.losses["Ls_D_txt_rl"].update(errD_txt_real.data[0], self.args.batch_size)
        # self.losses["Ls_GP_txt"].update(errGP_txt.data[0], self.args.batch_size)
            if self.double_fake_error:
                self.losses["Ls_D_txt_wr"].update(errD_txt_wrong.data[0], self.args.batch_size)

        # Gradient Penalty

       # img_gp = calc_gradient_penalty(self.networks["discriminator"], images, img_gen)


    def train_G(self, epoch, images, captions, lengths, pretrain = False, train_img = True, train_txt = True):

        # Forward, Backward and Optimize
        self.networks_zero_grad()

        if self.use_variational:
            img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_z, txt_z, img_mu, img_logv, txt_mu, txt_logv = \
                                 self.networks["coupled_vae"](images, captions, lengths)
        else:
            img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_z, txt_z =\
                self.networks["coupled_vae"](images, captions, lengths)

        img_rc_loss = 0
        txt_rc_loss = 0
        if pretrain or self.use_parallel:

            if self.use_variational:
                if train_img:
                    img_rc_loss = img_vae_loss(img2img_out, images, img_mu, img_logv) /\
                               (self.args.batch_size * (self.args.crop_size**2))


                if train_txt:
                    NLL_loss_txt, NLL_loss_img, KL_loss, KL_weight = seq_vae_loss(txt2txt_out, img2txt_out, captions,
                                                       lengths, txt_mu, txt_logv, "logistic", self.step, 0.0025,
                                                       2500)
                    txt_rc_loss = (NLL_loss_img + KL_weight * KL_loss) / torch.sum(lengths).float()
            else:

                if train_img:
                    img_rc_loss = self.networks["coupled_vae"].image_reconstruction_loss(images, img2img_out)

                if train_txt:
                    txt_rc_loss = self.networks["coupled_vae"].text_reconstruction_loss(captions, txt2txt_out, lengths)

            vae_loss = txt_rc_loss + img_rc_loss
            self.step += 1

            # Only retain graph if not on pretraining mode
            vae_loss.backward(retain_graph = not pretrain)

        if not pretrain:
            gen_len = get_gen_lengths(img2txt_out)
            if self.cycle_consistency:
                cc_img_mu, cc_img_logv, cc_img_z, cc_txt_mu, cc_txt_logv, cc_txt_z = \
                    self.networks["coupled_vae"].reconstruct(txt2img_out, img2txt_out, gen_len)

                lambda_CC = min(1.0,0.01 * epoch)
                cc_loss_img = mse_loss(cc_img_z[:,:self.masked_latent_size], img_z[:,:self.masked_latent_size].detach())
                cc_loss_txt = mse_loss(cc_txt_z[:,:self.masked_latent_size], txt_z[:,:self.masked_latent_size].detach())


                cc_loss = lambda_CC * (cc_loss_img + cc_loss_txt)
                cc_loss.backward(retain_graph = True)

            lambda_G = 1

            if train_txt:
                errG_txt, = self.networks["txt_discriminator"](img2txt_out, img_z, gen_len)[0] * lambda_G
                errG_txt.backward(self.one)

            if train_img:
                txt2img_out = add_gaussian(txt2img_out, std=0.05 * (self.args.num_epochs - epoch)/self.args.num_epochs)
                images = add_gaussian(images, std=0.05 * (self.args.num_epochs - epoch)/self.args.num_epochs)
                errD_img_real, act_real = self.networks["img_discriminator"](images, txt_z.detach())
                errG_img, act_fake = self.networks["img_discriminator"](txt2img_out, txt_z) * lambda_G
                err_img = errG_img

                if self.use_feature_matching:
                    errFM_img = get_fm_loss(act_real, act_fake)
                    err_img += errFM_img

                err_img.backward(self.one)

        self.optimizers["coupled_vae"].step()

        ### Log Losses
        if not pretrain:
            if train_img:
                self.losses["Ls_G_img"].update(errG_img.data[0], self.args.batch_size)
            if train_txt:
                self.losses["Ls_G_txt"].update(errG_txt.data[0], self.args.batch_size)

            if self.cycle_consistency:
                self.losses["Ls_CC_img"].update(cc_loss_img.data[0], self.args.batch_size)
                self.losses["Ls_CC_txt"].update(cc_loss_txt.data[0], self.args.batch_size)

        if pretrain or self.use_parallel:
            if train_img:
                self.losses["Ls_RC_img"].update(img_rc_loss.data[0], self.args.batch_size)

            if train_txt:
                self.losses["Ls_RC_txt"].update(txt_rc_loss.data[0], self.args.batch_size)

            self.losses["Ls_VAE"].update(vae_loss.data[0], self.args.batch_size)

    # def validate(self, loader):


    def evaluate(self, epoch, images, captions, lengths, save):

        if self.use_variational:
            img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_z, txt_z, img_mu, img_logv, txt_mu, txt_logv = \
                                 self.networks["coupled_vae"](images, captions, lengths)


            NLL_loss, _, KL_loss, KL_weight = seq_vae_loss(txt2txt_out, img2txt_out, captions,
                                                   lengths, txt_mu, txt_logv, "logistic", self.step, 0.0025,
                                                   2500)
        else:

            img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_z, txt_z =\
                self.networks["coupled_vae"](images, captions, lengths)

            img_rc_loss = self.networks["coupled_vae"].image_reconstruction_loss(images, img2img_out)
            # txt_rc_loss = self.networks["coupled_vae"].text_reconstruction_loss(captions, txt2txt_out, lengths)
            txt_rc_loss = self.networks["coupled_vae"].text_reconstruction_loss(captions, txt2txt_out, lengths)



        perm_idz = torch.randperm(self.args.batch_size).cuda()
        gen_lengths = get_gen_lengths(img2txt_out)

        NLL_loss = NLL_loss/ torch.sum(gen_lengths).float()

        vae = self.networks["coupled_vae"]
        txtL2 = mse_loss(vae.Z2Hidden_txt4txt(txt_z), vae.Z2Hidden_img4txt(img_z))
        errD_txt_fake = self.networks["txt_discriminator"](img2txt_out.detach(), img_z.detach(), gen_lengths)[0]
        errD_txt_real, _ = self.networks["txt_discriminator"](captions.detach(), img_z.detach(), lengths)
        errD_txt_wrong = self.networks["txt_discriminator"](captions.detach(), img_z[perm_idz].detach(), lengths)[0]

        errD_img_real, act_real = self.networks["img_discriminator"](images.detach(), txt_z.detach())
        errD_img_fake, act_fake = self.networks["img_discriminator"](txt2img_out.detach(), txt_z.detach())
        errD_img_wrong = self.networks["img_discriminator"](images.detach(), txt_z[perm_idz].detach())[0]
        #
        # # Compute BLEU score
        # pred_sents = []
        # trg_sents = []
        # for j in range(self.args.batch_size):
        #     txt_or = " ".join([self.vocab.idx2word[c] for c in caption.cpu().data.numpy()])
        #     txt_or = " ".join([self.vocab.idx2word[c] for c in generated[:,0].cpu().data.numpy()])
        #
        #     _, generated = torch.topk(txt_out,1)
        #     txt = " ".join([self.vocab.idx2word[c] for c in generated[:,0].cpu().data.numpy()])
        #
        #     pred_sent = get_sentence((img2txt_out[j].data.cpu().numpy().argmax(axis=-1), 'trg')
        #     trg_sent = get_sentence((captions[j].data.cpu().numpy().argmax(axis=1))), 'trg')
        #     pred_sents.append(pred_sent)
        #     trg_sents.append(trg_sent)
        # # bleu_value = get_bleu(pred_sents, trg_sents)


        self.metrics["Ls_RC_txt"].update(NLL_loss.data[0], self.args.batch_size)

        self.metrics["TxtL2"].update(txtL2.data[0], self.args.batch_size)

        self.metrics["Ls_D_img_fk"].update(errD_img_fake.data[0], self.args.batch_size)
        self.metrics["Ls_D_img_rl"].update(errD_img_real.data[0], self.args.batch_size)
        self.metrics["Ls_D_img_wr"].update(errD_img_wrong.data[0], self.args.batch_size)

        self.metrics["Ls_D_txt_fk"].update(errD_txt_fake.data[0], self.args.batch_size)
        self.metrics["Ls_D_txt_rl"].update(errD_txt_real.data[0], self.args.batch_size)
        self.metrics["Ls_D_txt_wr"].update(errD_txt_wrong.data[0], self.args.batch_size)


        if save:
            self.save_samples(img2img_out[0], txt2img_out[0], txt2txt_out[0], img2txt_out[0], False)

