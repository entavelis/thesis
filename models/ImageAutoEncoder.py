import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(
            self,
            img_dimension=256,
            feature_dimension = 300,
            reluBeta = 0.3
            ):

        super(ImageEncoder, self).__init__()

        self.feat_dim = feature_dimension

        if img_dimension == 64 :
            self.main = nn.Sequential(
                nn.Conv2d(3, img_dimension, 4, 2, 1, bias=False),
                nn.LeakyReLU(reluBeta, inplace=True),
                nn.Conv2d(img_dimension, img_dimension * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img_dimension * 2),
                nn.LeakyReLU(reluBeta, inplace=True),
                nn.Conv2d(img_dimension * 2, img_dimension * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img_dimension * 4),
                nn.LeakyReLU(reluBeta, inplace=True),
                nn.Conv2d(img_dimension * 4, img_dimension * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img_dimension * 8),
                nn.LeakyReLU(reluBeta, inplace=True),
                # nn.Conv2d(img_dimension * 8, img_dimension * 8, 4, 2, 1, bias=False),
                # nn.BatchNorm2d(img_dimension * 8),
                # nn.LeakyReLU(reluBeta, inplace=True),
                nn.Conv2d(img_dimension * 8, feature_dimension, 4, 1, 0, bias=False),
                nn.BatchNorm2d(feature_dimension),
                nn.LeakyReLU(reluBeta, inplace=True),
            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(3, img_dimension, 4, 2, 1, bias=False),
                nn.LeakyReLU(reluBeta, inplace=True),
                nn.Conv2d(img_dimension, img_dimension * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img_dimension * 2),
                nn.LeakyReLU(reluBeta, inplace=True),
                nn.Conv2d(img_dimension * 2, img_dimension * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img_dimension * 4),
                nn.LeakyReLU(reluBeta, inplace=True),
                nn.Conv2d(img_dimension * 4, img_dimension * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img_dimension * 8),
                nn.LeakyReLU(reluBeta, inplace=True),
                nn.Conv2d(img_dimension * 8, img_dimension * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img_dimension * 8),
                nn.LeakyReLU(reluBeta, inplace=True),
                nn.Conv2d(img_dimension * 8, feature_dimension, 4, 1, 0, bias=False),
                nn.BatchNorm2d(feature_dimension),
                nn.LeakyReLU(reluBeta, inplace=True),
            )

    def forward(self, input):
        x = self.main(input)
        # x_cap = x.view(-1, self.feat_dim * 4 * 4)
        # x_cap = self.fc(x_cap)
        return x, x.view(-1, self.feat_dim)


class ImageDecoder(nn.Module):
    def __init__(
            self,
            img_dimension=256,
            feature_dimension =300,
            reluBeta = 0.2
            ):

        super(ImageDecoder, self).__init__()

        if img_dimension == 64:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(feature_dimension, img_dimension * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(img_dimension * 8),
                nn.LeakyReLU(reluBeta, inplace=True),
                # nn.ConvTranspose2d(img_dimension*8, img_dimension * 8, 4, 2, 1, bias=False),
                # nn.BatchNorm2d(img_dimension * 8),
                # nn.ReLU(True),
                nn.ConvTranspose2d(img_dimension * 8, img_dimension * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img_dimension * 4),
                nn.LeakyReLU(reluBeta, inplace=True),
                nn.ConvTranspose2d(img_dimension * 4, img_dimension * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img_dimension * 2),
                nn.LeakyReLU(reluBeta, inplace=True),
                nn.ConvTranspose2d(img_dimension * 2,     img_dimension, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img_dimension),
                nn.LeakyReLU(reluBeta, inplace=True),
                nn.ConvTranspose2d(img_dimension,      3, 4, 2, 1, bias=False),
                # nn.Sigmoid()
            )
        else:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(feature_dimension, img_dimension * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(img_dimension * 8),
                nn.LeakyReLU(reluBeta, inplace=True),
                nn.ConvTranspose2d(img_dimension*8, img_dimension * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img_dimension * 8),
                nn.LeakyReLU(reluBeta, inplace=True),
                nn.ConvTranspose2d(img_dimension * 8, img_dimension * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img_dimension * 4),
                nn.LeakyReLU(reluBeta, inplace=True),
                nn.ConvTranspose2d(img_dimension * 4, img_dimension * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img_dimension * 2),
                nn.LeakyReLU(reluBeta, inplace=True),
                nn.ConvTranspose2d(img_dimension * 2,     img_dimension, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img_dimension),
                nn.LeakyReLU(reluBeta, inplace=True),
                nn.ConvTranspose2d(img_dimension,      3, 4, 2, 1, bias=False),
                nn.Tanh()
            )

    def forward(self, input):
        return self.main( input )