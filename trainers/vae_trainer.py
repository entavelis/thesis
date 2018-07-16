import os
import torch.optim as optim

from models.ImgVAE import ImgVAE
from models.SeqVAE import SentenceVAE

from losses import *
from trainers.trainer import trainer
from utils import *

class coupled_vae_trainer(trainer):

    def __init__(self, args, embedding, vocab):
        super(coupled_vae_trainer,self).__init__(args, embedding, vocab)

        vae_Txt = SentenceVAE(embedding, len(vocab), hidden_size=args.hidden_size, latent_size=args.latent_size,
                         batch_size = args.batch_size)

        vae_Img = ImgVAE(img_dimension=args.crop_size, hidden_size=args.hidden_size, latent_size= args.latent_size)

        img_optim = optim.Adam(vae_Img.parameters(), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)
        txt_optim = optim.Adam(vae_Txt.parameters(), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)

        self.networks["seq_vae"] = vae_Txt
        self.networks["img_vae"] = vae_Img


        self.optimizers["seq_vae"] = txt_optim
        self.optimizers["img_vae"] = img_optim

        self.nets_to_cuda()

        self.train_swapped = False # Reverse 2
        self.step = 0

    def train(self, epoch, images, captions, lengths, save_images):

        self.zero_grad()
        # Forward, Backward and Optimize
        img_out, img_mu, img_logv, img_z = self.networks["img_vae"](images)
        txt_out, txt_mu, txt_logv, txt_z = self.networks["seq_vae"](captions, lengths)

        img_rc_loss = img_vae_loss(img_out, images, img_mu, img_logv) / (self.args.batch_size * self.args.crop_size**2)

        NLL_loss, KL_loss, KL_weight = seq_vae_loss(txt_out, captions,
                                               lengths, txt_mu, txt_logv, "logistic", self.step, 0.0025,
                                               2500)

        txt_rc_loss = (NLL_loss + KL_weight * KL_loss) / torch.sum(lengths).float()

        cm_loss = crossmodal_loss(txt_z, img_z, self.mask,
                                  self.args.cm_criterion, self.cm_criterion,
                                      self.args.negative_samples, epoch)

        if save_images:
            self.save_samples(images[0], img_out[0], captions[0], txt_out[0])
        return img_rc_loss, txt_rc_loss, cm_loss

    def backpropagate(self, loss, train_image):
        if train_image:
            loss.backward()

        self.step += 1

