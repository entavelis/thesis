import torch
import torch.nn as nn

class ImgDiscriminator(nn.Module):
    def __init__(
            self,
            img_dimension = 64,
            betaReLU = 0.2,
            latent_size = 512,
            mask = 0.0,
            batch_size = 32
            ):

        super(ImgDiscriminator, self).__init__()
        self.batch_size = batch_size
        self.masked_size = int(latent_size * mask)
        if mask>0:
            self.seq_embedding = nn.Linear(self.masked_size, 128*4*4)

        self.conv1 = nn.Conv2d(3, img_dimension, 4, 2, 1, bias=False)
        self.relu1 = nn.LeakyReLU(betaReLU, inplace=True)

        self.conv2 = nn.Conv2d(img_dimension, img_dimension * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(img_dimension * 2)
        self.relu2 = nn.LeakyReLU(betaReLU, inplace=True)

        self.conv3 = nn.Conv2d(img_dimension * 2, img_dimension * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(img_dimension * 4)
        self.relu3 = nn.LeakyReLU(betaReLU, inplace=True)

        self.conv4 = nn.Conv2d(img_dimension * 4, img_dimension * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(img_dimension * 8)
        self.relu4 = nn.LeakyReLU(betaReLU, inplace=True)

        self.conv5 = nn.Conv2d(img_dimension * 8 + (128 if mask>0 else 0), 1, 4, 1, 0, bias=False)


    def forward(self, image_input, seq_emb = None):
        conv1 = self.conv1( image_input )
        relu1 = self.relu1( conv1 )

        conv2 = self.conv2( relu1 )
        bn2 = self.bn2( conv2 )
        relu2 = self.relu2( bn2 )

        conv3 = self.conv3( relu2 )
        bn3 = self.bn3( conv3 )
        relu3 = self.relu3( bn3 )

        conv4 = self.conv4( relu3 )
        bn4 = self.bn4( conv4 )
        relu4 = self.relu4( bn4 )

        if self.masked_size > 0:
            seq = self.seq_embedding(seq_emb[:,:self.masked_size])
            augmented = torch.cat([relu4, seq.view(self.batch_size, 128,4,4)],1)
            conv5 = self.conv5(augmented)
        else:
            conv5 = self.conv5(relu4)

        return torch.sigmoid(conv5).mean(0).view(1) #, [relu2, relu3, relu4]

