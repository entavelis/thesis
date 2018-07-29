import os
import torch.optim as optim
from models.CoupledVAE import *
from losses import *
from trainers.trainer import trainer
from utils import *

class coupled_vae_trainer(trainer):

    def __init__(self, args, embedding, vocab):
        super(coupled_vae_trainer,self).__init__(args, embedding, vocab)

        self.networks["coupled_vae"] = CoupledVAE(embedding,
                                                  len(vocab),
                                                  hidden_size=args.hidden_size,
                                                  latent_size=args.latent_size,
                                                  batch_size = args.batch_size)


        self.optimizers["coupled_vae"] = optim.Adam(valid_params(self.networks["coupled_vae"].parameters()),\
                                                    lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)

        self.nets_to_cuda()

        self.train_swapped = False # Reverse 2
        self.step = 0

    def train(self, images, captions, lengths, save_images):

        # Forward, Backward and Optimize

        img_out, img_mu, img_logv, img_z, txt_out, txt_mu, txt_logv, txt_z = \
                                 self.networks["coupled_vae"](images, captions, lengths, self.train_swapped)

        img_rc_loss = img_vae_loss(img_out, images, img_mu, img_logv) /\
                      (self.args.batch_size * self.args.crop_size**2)

        NLL_loss, KL_loss, KL_weight = seq_vae_loss(txt_out, captions,
                                               lengths, txt_mu, txt_logv, "logistic", self.step, 0.0025,
                                               2500)
        txt_rc_loss = (NLL_loss + KL_weight * KL_loss) / torch.sum(lengths).float()

        return img_rc_loss, txt_rc_loss

    def backpropagate(self, loss):
        loss.backward()
        for opt in self.optimizers.values():
            opt.step()
        self.step += 1

