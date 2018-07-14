import torch
import os

class trainer( object ):
    """Some description that tells you it's abstract,
    often listing the methods you're expected to supply."""
    networks = {}
    optimizers = {}
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
            self.result_path = self.model_path + "results/"

        with open(os.path.join(self.result_path,"losses.csv") , "w") as text_file:
            text_file.write("Epoch, Img, Txt, CM\n")

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def train(self):
        raise NotImplementedError( "Should have implemented this" )

    def backpropagate(self):
        raise NotImplementedError( "Should have implemented this" )

    def load_models(self, epoch):
        if self.keep_loading:
            for name, network in self.networks:
                suffix = name + "-" + str(epoch) + ".pkl"
                try:
                    network.load_state_dict(torch.load(os.path.join(self.model_path, suffix)))

                except FileNotFoundError:
                    print("Didn't find any models switching to training")
                    self.keep_loading = False
                    return False

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

    def set_train_models(self):
        for net in self.networks.values():
            net.train()