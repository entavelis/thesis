#Source: https://github.com/timbmg/Sentence-VAE/blob/master/model.py
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var
from utils import gumbel_softmax
import torch.nn.functional as F
from torch.autograd import Variable

class CoupledVAE(nn.Module):
    def __init__(self,  embedding, vocab_size,
                img_dimension=256,
                hidden_size = 1024,
                latent_size = 512,
                num_layers=1,
                bidirectional = False,
                rnn_type = "GRU",
                word_dropout = 0.5,
                sos_idx = 1,
                eos_idx = 2,
                pad_idx = 0,
                max_sequence_length = 30,
                batch_size = 128,
                mask = 0.3,
                drop_out = 0.3,
                averaged_output = False):

        super(CoupledVAE, self).__init__()

        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        # like rnn-wgan
        self.averaged_output = True

        self.vocab_size = vocab_size

        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        self.mask = mask
        self.common_z_size = int(mask * latent_size)
        self.left_z_size = latent_size - self.common_z_size

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.img_dimension = img_dimension

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.embedding = embedding
        embedding_size = embedding.embedding_dim

        self.word_dropout = nn.Dropout(p=word_dropout)

        if rnn_type == 'RNN':
            rnn = nn.RNN
        elif rnn_type == 'GRU':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)

        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)



        self.encoder_cnn = nn.Sequential(
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

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_size, img_dimension * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(img_dimension * 8),
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
            # nn.Tanh()
        )

        self.hidden2mean = nn.Linear(self.hidden_size, latent_size)
        self.hidden2logv = nn.Linear(self.hidden_size, latent_size)
        self.latent2hidden = nn.Linear(latent_size, self.hidden_size)

        self.img_noise = torch.FloatTensor(self.batch_size, self.left_z_size).cuda()
        self.txt_noise = torch.FloatTensor(self.batch_size, self.left_z_size).cuda()


    def forward(self, input_images, input_captions, lengths):
        sorted_idx, self.packed_input = self.prepare_input(input_captions, lengths)

        img_enc = self.img_encoder_forward(input_images)
        txt_enc = self.txt_encoder_forward(self.packed_input)

        img_mu, img_logv, img_z = self.Hidden2Z(img_enc)
        txt_mu, txt_logv, txt_z = self.Hidden2Z(txt_enc)

        hidden4img2img = self.Z2Hidden(img_z)
        hidden4txt2txt = self.Z2Hidden(txt_z)

        img_noise = to_var(self.img_noise.resize_(self.batch_size, self.left_z_size).normal_(0,1))
        txt_noise = to_var(self.txt_noise.resize_(self.batch_size, self.left_z_size ).normal_(0,1))

        hidden4txt2img = self.Z2Hidden(torch.cat((txt_z[:,:self.common_z_size], img_noise),1))

        hidden4img2txt = self.Z2Hidden(torch.cat((img_z[:,:self.common_z_size], txt_noise),1))


        # for AE
        img2img_out = self.img_decoder_forward(hidden4img2img)
        # txt2txt_out = self.txt_decoder_forward(hidden4txt2txt) # try_decoder
        txt2txt_out = self.gumbel_decoder(hidden4txt2txt, input_captions.size(1)) # try_decoder

        # for discriminator
        txt2img_out = self.img_decoder_forward(hidden4txt2img)
        img2txt_out = self.gumbel_decoder(hidden4img2txt, input_captions.size(1))


        return img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_mu, img_logv, img_z, txt_mu, txt_logv, txt_z

    def Hidden2Z(self, hidden):
        mu = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)

        # REPARAMETERIZATION
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([self.batch_size, self.latent_size]))
        z = z * std + mu

        return mu, logv, z

    def Z2Hidden(self, z):
        return self.latent2hidden(z)

    def img_encoder_forward(self, input):

        enc = self.encoder_cnn(input)
        enc = enc.view(self.batch_size,-1)
        return enc

    def img_decoder_forward(self, input):
        output = self.decoder_cnn(input.view(self.batch_size, self.hidden_size, 1 , 1))
        return  output

    def prepare_input(self, input_sequence, length):
        # Remove
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]
        input_embedding = self.embedding(input_sequence)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        return  sorted_idx, packed_input

    def txt_encoder_forward(self, packed_input):
        # ENCODER
        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(self.batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        return hidden

    # DECODER
    def txt_decoder_forward(self,hidden, packed_input, sorted_idx):

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, self.batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        # input_embedding = self.word_dropout(input_embedding)
        # packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return logp


    def reconstruct(self, gen_images, gen_captions):

        img_enc = self.img_encoder_forward(gen_images)
        txt_enc = self.txt_encoder_forward(gen_captions)

        img_mu, img_logv, img_z = self.Hidden2Z(img_enc)
        txt_mu, txt_logv, txt_z = self.Hidden2Z(txt_enc)


        return img_mu, img_logv, img_z, txt_mu, txt_logv, txt_z

    def gumbel_decoder(self, hidden, max_sequence_length):

        batch_size = self.batch_size


        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        # generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()
        # generations = self.tensor(batch_size, max_sequence_length, self.vocab_size).fill_(self.pad_idx).float()
        generations = torch.zeros(batch_size, max_sequence_length, self.vocab_size).float()
        generations[:,:,0] = 1.0
        generations = to_var(generations)
        generations.requires_grad = True

        # generations.scatter_(1,0,1.0)

        t=0
        while(t<max_sequence_length and len(running_seqs)>0):

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())
                input_sequence = input_sequence.unsqueeze(1)

                input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = F.log_softmax(self.outputs2vocab(output),2)

            # input_sequence = self._sample(logits)
            input_sequence = gumbel_softmax(logits.squeeze()).unsqueeze(1)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update global running sequence
            _, input_sequence_top = torch.topk(input_sequence, 1)
            sequence_mask[sequence_running] = (input_sequence_top.squeeze() != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence_top.squeeze() != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]
                input_embedding = torch.mm(input_sequence.squeeze(), self.embedding.weight).unsqueeze(1)

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample # .data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
