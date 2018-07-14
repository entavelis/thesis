import torch
import torch.nn as nn

class SeqDiscriminator(nn.Module):
    def __init__(
            self,
            seq_dimension = 30
            ):

        super(SeqDiscriminator, self).__init__()
        self.conv1 = nn.Conv1d(2, seq_dimension, 4, 2, 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv1d(seq_dimension, seq_dimension * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(seq_dimension * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv1d(seq_dimension * 2, seq_dimension * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(seq_dimension * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv1d(seq_dimension * 4, seq_dimension * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm1d(seq_dimension * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv1d(seq_dimension * 8, 1, 4, 1, 0, bias=False)

    def forward(self, input):
        conv1 = self.conv1( input )
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

        conv5 = self.conv5( relu4 )

        return torch.sigmoid( conv5 ), [relu2, relu3, relu4]

