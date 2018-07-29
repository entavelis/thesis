
import torch
import os
import scipy
from pytorch_classification.utils import AverageMeter
import csv

class trainer( object ):
    """Some description that tells you it's abstract,
    often listing the methods you're expected to supply."""
    networks = {}
    optimizers = {}
    losses = {}
    cuda = True

    def __init__( self, args, embeddings, vocab):
        self.args = args
        self.embeddings = embeddings
        self.vocab = vocab

        if args.load_model == "NONE":
            self.keep_loading = False
            self.model_path = args.model_path + args.comment + "/"
        else:
            self.keep_loading = True
            self.model_path = args.model_path + args.load_model + "/"

        result_path = args.result_path
        if result_path == "NONE":
            self.result_path = self.model_path + "samples/"


        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.mask = int(args.common_emb_ratio * args.hidden_size)

        self.save_counter = 1
        self.iteration = 0

        self.save_options()

    def train(self):
        raise NotImplementedError( "Should have implemented this" )

    def backpropagate(self):
        raise NotImplementedError( "Should have implemented this" )

    def load_models(self, epoch):
        if self.keep_loading:
            for name, network in self.networks.items():
                suffix = name + "-" + str(epoch) + ".pkl"
                try:
                    network.load_state_dict(torch.load(os.path.join(self.model_path, suffix)))

                except FileNotFoundError:
                    print("Didn't find any models switching to training")
                    self.keep_loading = False
                    return False

            print("Model loaded for epoch ", epoch)
            return True
        return False

    def save_models(self, epoch):
        # Save the models
        print('\n')
        print('Saving the models in {}...'.format(self.model_path))
        for name, net in self.networks.items():
            suffix = name + "-" + str(epoch) + ".pkl"
            torch.save(net.state_dict(), os.path.join(self.model_path, suffix))

    def nets_to_cuda(self):
        if self.cuda:
            for i,n in self.networks.items():
                self.networks[i] = n.cuda()


    def networks_zero_grad(self):
        for n in self.networks.values():
            n.zero_grad()

    def set_eval_models(self):
        for net in self.networks.values():
            net.eval()

    def set_train_models(self):
        for net in self.networks.values():
            net.train()

    def save_samples(self, image, img_out, caption, txt_out):
        # subdir_path = os.path.join(self.result_path, str(self.save_counter))
        subdir_path = self.result_path

        if os.path.exists(subdir_path):
            pass
        else:
            os.makedirs(subdir_path)

        # im_or = (images[im_idx].cpu().data.numpy().transpose(1,2,0))*255
        # im = (img_out[im_idx].cpu().data.numpy().transpose(1,2,0))*255
        im_or = (image.cpu().data.numpy().transpose(1, 2, 0) / 2 + .5) * 255
        im = (img_out.cpu().data.numpy().transpose(1, 2, 0) / 2 + .5) * 255
        # im = img_out[im_idx].cpu().data.numpy().transpose(1,2,0)*255

        filename_prefix = os.path.join(subdir_path, str(self.save_counter).zfill(3))
        scipy.misc.imsave(filename_prefix + '_original.jpg', im_or)
        scipy.misc.imsave(filename_prefix + '.jpg', im)

        try:
            txt_or = " ".join([self.vocab.idx2word[c] for c in caption.cpu().data.numpy()])
        except:
            _, generated = torch.topk(caption,1)
            txt_or = " ".join([self.vocab.idx2word[c] for c in generated[:,0].cpu().data.numpy()])

        try:
            _, generated = torch.topk(txt_out,1)
            txt = " ".join([self.vocab.idx2word[c] for c in generated[:,0].cpu().data.numpy()])

            with open(filename_prefix + "_captions.txt", "w") as text_file:
                text_file.write("Save_counter %d\n" % self.save_counter)
                text_file.write("Original:\t %s\n" % txt_or)
                text_file.write("Generated:\t %s" % txt)
        except:
            pass



        self.save_losses()
        self.save_counter += 1

    # change names
    def save_losses(self):
        with open(os.path.join(self.result_path,"losses.csv") , "a") as text_file:
            toWrite= str(self.iteration)
            for l_value in self.losses.values():
                toWrite += ' , {}'.format(l_value.avg)
            text_file.write(toWrite + "\n")


    def create_losses_meter(self, losses_name_list):
        toWrite = "Iteration"
        for name in losses_name_list:
            self.losses[name] = AverageMeter()

        for name in self.losses.keys():
            toWrite += ", " + name
        toWrite +="\n"

        with open(os.path.join(self.result_path,"losses.csv") , "w") as text_file: text_file.write(toWrite)

    def save_options(self):
        temp = self.args.__dict__
        with open(os.path.join(self.result_path,"arguments.csv") , 'w') as f:  # Just use 'w' mode in 3.x
            w = csv.writer(f)
            # w.writeheader()
            w.writerows(temp.items())
