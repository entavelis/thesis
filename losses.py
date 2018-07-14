import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def mse_loss(input, target, sim=None):
    if sim is None:
        return torch.sum((input - target)**2) / input.data.nelement()
    else:
        return torch.mean(sim*torch.mean((input - target)**2,1))

# def cosine_loss(input,target,sim=None);

# source: https://github.com/howardyclo/pytorch-seq2seq-example/blob/master/seq2seq.ipynb
def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.

    The code is same as:

    weight = torch.ones(tgt_vocab_size)
    weight[padding_idx] = 0
    criterion = nn.CrossEntropyLoss(weight.cuda(), size_average)
    loss = criterion(logits_flat, losses_flat)
    """
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    # Note: mask need to bed casted to float!
    losses = losses * mask.float()
    loss = losses.sum() / mask.float().sum()

    # (batch_size * max_tgt_len,)
    pred_flat = log_probs_flat.max(1)[1]
    # (batch_size * max_tgt_len,) => (batch_size, max_tgt_len) => (max_tgt_len, batch_size)
    pred_seqs = pred_flat.view(*target.size()).transpose(0, 1).contiguous()
    # (batch_size, max_len) => (batch_size * max_tgt_len,)
    mask_flat = mask.view(-1)

    # `.float()` IS VERY IMPORTANT !!!
    # https://discuss.pytorch.org/t/batch-size-and-validation-accuracy/4066/3
    num_corrects = int(pred_flat.eq(target_flat.squeeze(1)).masked_select(mask_flat).float().data.sum())
    num_words = length.data.sum()

    return loss, pred_seqs, num_corrects, num_words

# Reconstruction + KL divergence losses summed over all elements and batch
def img_vae_loss(recon_x, x, mu, logvar):
    # print(recon_x.size(), x.size())
    flat_dim = x.size(2)**2
    flat_rc_x = recon_x.view(-1, flat_dim)
    flat_x = x.view(-1, flat_dim)

    # BCE = F.binary_cross_entropy(flat_rc_x, flat_x, size_average=False)
    BCE = F.mse_loss(flat_rc_x, flat_x, size_average=False)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # return BCE + KLD
    return (BCE + 3 * KLD)



def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

def seq_vae_loss(logp, target, length, mean, logv, anneal_function, step, k, x0):
        # cut-off unnecessary padding from target, and flatten
    NLL = torch.nn.NLLLoss(size_average=False, ignore_index=0)

    target = target[:, :torch.max(length).data[0]].contiguous().view(-1)
    logp = logp.view(-1, logp.size(2))

    # Negative Log Likelihood
    NLL_loss = NLL(logp, target)

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    KL_weight = kl_anneal_function(anneal_function, step, k, x0)

    return NLL_loss, KL_loss, KL_weight

def crossmodal_loss(txt_z, img_z, mask, cm_type, cm_criterion, negative_samples, epoch):
    # txt = txt_z.narrow(1,0,mask)
    # im = img_z.narrow(1,0,mask)
    txt = txt_z[:,:mask]
    im = img_z[:,:mask]

    batch_size = img_z.size(0)
    cuda = torch.cuda.is_available()

    if cm_type == 'MSE':
        # cm_loss = cm_criterion(Tz.narrow(1,0,mask), Iz.narrow(1,0,mask))
        cm_loss = mse_loss(txt, im)
    else:
        cm_loss = cm_criterion(txt, im, \
            Variable(torch.ones(batch_size).cuda()))

    # K - Negative Samples
    k = negative_samples
    neg_rate = (20-epoch)/40
    for _ in range(k):

        if cuda:
            perm = torch.randperm(batch_size).cuda()
        else:
            perm = torch.randperm(batch_size)

        # if args.criterion == 'MSE':
        #     cm_loss -= mse_loss(txt, im[perm])/k
        # else:
        #     cm_loss -= cm_criterion(txt, im[perm], \
        #                            Variable(torch.ones(Tz.narrow(1,0,mask).size(0)).cuda()))/k

        # sim  = (F.cosine_similarity(txt,txt[perm]) - 0.5)/2

        if cm_type == 'MSE':
            sim  = (F.cosine_similarity(txt,txt[perm]) - 1)/(2*k)
            # cm_loss = cm_criterion(Tz.narrow(1,0,mask), Iz.narrow(1,0,mask))
            cm_loss += mse_loss(txt, im[perm], sim)
        else:
            cm_loss += neg_rate * cm_criterion(txt, im[perm], \
            Variable(-1*torch.ones(batch_size).cuda()))/k

    # cm_loss = Variable(torch.max(torch.FloatTensor([-0.100]).cuda(), cm_loss.data))
    return cm_loss

