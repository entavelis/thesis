import torch
import torch.nn as nn
from utils import to_var

class ImgVAE(nn.Module):
    def __init__(
            self,
            img_dimension=256,
            feature_dimension = 300
            ):

        super(ImgVAE, self).__init__()

        self.feat_dim = feature_dimension
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
            nn.Conv2d(img_dimension * 8, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 64 * 8, 4, 1, 0, bias=False),
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

        self.hidden2mean = nn.Linear(1024, feature_dimension)
        self.hidden2logv = nn.Linear(1024, feature_dimension)

        self.latent2hidden = nn.Linear(feature_dimension, 1024)


    def forward(self, input):
        batch_size = input.size(0)

        enc = self.encoder(input)
        enc = enc.view(batch_size,-1)

        mu = self.hidden2mean(enc)
        logv = self.hidden2logv(enc)

        # REPARAMETERIZATION
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.feat_dim]))
        z = z * std + mu

        dec_input = self.latent2hidden(z)

        output = self.decoder(dec_input.view(batch_size, 1024, 1 , 1))

        return output, mu, logv, z


