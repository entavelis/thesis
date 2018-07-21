import torch
import torch.nn as nn
from utils import to_var

class ImgVAE(nn.Module):
    def __init__(
            self,
            img_dimension=256,
            hidden_size = 1024,
            latent_size = 512
            ):

        super(ImgVAE, self).__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.img_dimension = img_dimension

        self.encoder = nn.Sequential(
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

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_size, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(img_dimension * 8, img_dimension * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_dimension * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(img_dimension * 4, img_dimension * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_dimension * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(img_dimension * 2,     img_dimension, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_dimension),
            nn.ReLU(True),
            nn.ConvTranspose2d(img_dimension,      3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.hidden2mean = nn.Linear(self.hidden_size, latent_size)
        self.hidden2logv = nn.Linear(self.hidden_size, latent_size)

        self.latent2hidden = nn.Linear(latent_size, self.hidden_size)

    def forward(self, input):
        mu, logv, z = self.encoder_forward(input)
        output = self.decoder_forward(z)
        return output, mu, logv, z

    def encoder_forward(self, input):
        batch_size = input.size(0)

        enc = self.encoder(input)
        enc = enc.view(batch_size,-1)

        mu = self.hidden2mean(enc)
        logv = self.hidden2logv(enc)

        # REPARAMETERIZATION
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mu

        return mu, logv, z

    def decoder_forward(self, z):
        batch_size = z.size(0)
        dec_input = self.latent2hidden(z)

        output = self.decoder(dec_input.view(batch_size, self.hidden_size, 1 , 1))
        return  output


