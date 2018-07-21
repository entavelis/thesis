import os
import torch.optim as optim
from models.ImageAutoEncoder import ImageDecoder
from models.ImgDiscriminator import *

from losses import *
from trainers.trainer import trainer
from utils import *
from itertools import chain

class wgan_trainer(trainer):

    def __init__(self, args, embedding, vocab):
        super(wgan_trainer, self).__init__(args, embedding, vocab)

        # Setting up the networks
        self.networks["generator"] = ImageDecoder(feature_dimension=args.latent_size,
                                                  img_dimension= args.crop_size)

        self.networks["discriminator"] = ImgDiscriminator(args.crop_size)


        # Setting up the optimizers
        self.optimizers["generator"] = optim.Adam(self.networks["generator"].parameters(),\
                                                    lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)

        # self.optimizers["discriminator"] = optim.RMSprop(self.networks["discriminator"].parameters(),
        #                                                  lr=args.learning_rate)
        self.optimizers["discriminator"] = optim.Adam(self.networks["discriminator"].parameters(),\
                                                    lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)

        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1
        if args.cuda:
            self.one = self.one.cuda()
            self.mone = self.mone.cuda()


        # self.gen_loss = Variable(torch.FloatTensor([1]* args.batch_size))
        # self.dis_loss = Variable(torch.FloatTensor([1]* args.batch_size))
        self.nets_to_cuda()

        self.step = 0

        # Setting up the losses
        self.create_losses_meter(["Ls_D", "Ls_G", "Ls_D_rl", "Ls_D_fk","Ls_GP"])

        # Setting up the noise
        self.noise = torch.FloatTensor(args.batch_size, args.latent_size, 1, 1).cuda()
        self.fixed_noise = Variable(torch.FloatTensor(args.batch_size, args.latent_size, 1, 1).normal_(0, 1).cuda())

    def forward(self, epoch, images, captions, lengths, save_images):

        if self.iteration < 2500:
            cycle = 101
        else:
            cycle = 6

        # train_gen = self.iteration > 500 and self.iteration % 6
        if not self.iteration % cycle:
             #
            for p in self.networks["discriminator"].parameters():  # reset requires_grad
                p.requires_grad = False  # they are set to False below in netG update

            self.train_G(epoch, images, captions, lengths)
            self.optimizers["generator"].step()
            #
            for p in self.networks["discriminator"].parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
        else:

            self.train_D(epoch, images, captions, lengths)

        self.iteration += 1
        # Log Losses:
        # self.losses["GEN_loss"].update(self.gen_loss.data[0],self.args.batch_size)
        # self.losses["DIS_loss"].update(self.dis_loss.data[0],self.args.batch_size)

        if save_images:
            self.save_samples(images[0], self.networks["generator"](self.fixed_noise)[0], captions[0], captions[0])



    def train_D(self,epoch, images, captions, lengths):

        # clamp parameters to a cube
        for p in self.networks["discriminator"].parameters():
            p.data.clamp_(-0.01, 0.01)

        # train with real
        self.networks["discriminator"].zero_grad()

        errD_real = self.networks["discriminator"](images)
        errD_real.backward(self.one)

        # train with fake
        self.noise.resize_(self.args.batch_size, self.args.latent_size, 1, 1).normal_(0, 1)
        noisev = Variable(self.noise, volatile=True)  # totally freeze netG
        fake = Variable(self.networks["generator"](noisev).data)

        inputv = fake
        errD_fake = self.networks["discriminator"](inputv)
        errD_fake.backward(self.mone)

        errGP = calc_gradient_penalty(self.networks["discriminator"], images, fake)
        errGP.backward()

        errD = errD_real - errD_fake + errGP
        self.optimizers["discriminator"].step()

        self.losses["Ls_D"].update(errD.data[0], self.args.batch_size)
        self.losses["Ls_D_fk"].update(errD_fake.data[0], self.args.batch_size)
        self.losses["Ls_D_rl"].update(errD_real.data[0], self.args.batch_size)
        self.losses["Ls_GP"].update(errGP.data[0], self.args.batch_size)

        # Gradient Penalty

       # img_gp = calc_gradient_penalty(self.networks["discriminator"], images, img_gen)


    def train_G(self,epoch, images, captions, lengths):
        self.networks["generator"].zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        self.noise.resize_(self.args.batch_size, self.args.latent_size, 1, 1).normal_(0, 1)
        noisev = Variable(self.noise)
        fake = self.networks["generator"](noisev)
        errG = self.networks["discriminator"](fake)
        errG.backward(self.one)
        self.optimizers["generator"].step()

        self.losses["Ls_G"].update(errG.data[0], self.args.batch_size)

    def backpropagate(self, loss):
        loss.backward()
        for opt in self.optimizers.values():
            opt.step()
        self.step += 1


