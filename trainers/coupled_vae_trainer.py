import os
import torch.optim as optim
from models.CoupledVAE import *
from losses import *
from trainers.trainer import trainer
from utils import *

class coupled_vae_trainer(trainer):

    def __init__(self, args, embedding, vocab):
        super(coupled_vae_trainer,self).__init__(args, embedding, vocab)
        # Setting up the networks
        self.use_variational = args.variational == "true"
        self.networks["coupled_vae"] = CoupledVAE(embedding,
                                                  len(vocab),
                                                  hidden_size=args.hidden_size,
                                                  latent_size=args.latent_size,
                                                  batch_size = args.batch_size,
                                                  img_dimension= args.crop_size,
                                                  mask= args.common_emb_ratio,
                                                  use_variational= args.variational)


        self.optimizers["coupled_vae"] = optim.Adam(valid_params(self.networks["coupled_vae"].parameters()),\
                                                    lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)

        self.nets_to_cuda()

        self.step = 0

        self.create_losses_meter(["Ls_IMG", "Ls_SEQ"])

    def forward(self, epoch, images, captions, lengths, save_images):

        # Forward, Backward and Optimize
        self.networks_zero_grad()
        if not self.use_variational:
            img2img_out, txt2img_out, img2txt_out, txt2txt_out= \
                                     self.networks["coupled_vae"](images, captions, lengths)

            img_rc_loss = self.networks["coupled_vae"].image_reconstruction_loss(img2img_out, images)
            txt_rc_loss = self.networks["coupled_vae"].text_reconstruction_loss(txt2txt_out, captions, lengths)
            loss = 0.1*txt_rc_loss + img_rc_loss

        else:
            img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_mu, img_logv, img_z, txt_mu, txt_logv, txt_z= \
                                     self.networks["coupled_vae"](images, captions, lengths)

            img_rc_loss = img_vae_loss(img2img_out, images, img_mu, img_logv) /\
                           (self.args.batch_size * self.args.crop_size**2)

            NLL_loss, KL_loss, KL_weight = seq_vae_loss(txt2txt_out, captions,
                                                   lengths, txt_mu, txt_logv, "logistic", self.step, 0.0025,
                                                   2500)
            txt_rc_loss = (NLL_loss + KL_weight * KL_loss) / torch.sum(lengths).float()

            loss = txt_rc_loss + img_rc_loss


        self.backpropagate(loss)

        self.losses["Ls_IMG"].update(img_rc_loss.data[0], self.args.batch_size)
        self.losses["Ls_SEQ"].update(txt_rc_loss.data[0], self.args.batch_size)
        if save_images:
            self.save_samples(images[0], img2img_out[0], captions[0], txt2txt_out[0])


        return img_rc_loss, txt_rc_loss

    def backpropagate(self, loss):
        loss.backward()
        for opt in self.optimizers.values():
            opt.step()
        self.step += 1

