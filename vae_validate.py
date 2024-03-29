import time

from old.model import *

from utils import *

from pytorch_classification.utils import Bar, AverageMeter

from sklearn.neighbors import NearestNeighbors

import os

def validate(coupled_vae, loader, mask, limit = 1000, metric = "cosine"):

        # cm_criterion = nn.CosineEmbeddingLoss()
        cm_criterion = nn.MSELoss()

        # VALIDATION TIME
        print('\033[92mEPOCH ::: VALIDATION ::: ' )

        # Set Evaluation Mode
        coupled_vae.eval()

        batch_time = AverageMeter()
        end = time.time()

        bar = Bar('Computing Validation Set Embeddings', max=len(loader))

        cm_losses = AverageMeter()

        for i, (images, captions, lengths) in enumerate(loader):
            if i == limit:
                break

            # Set mini-batch dataset
            images = to_var(images)
            captions = to_var(captions)

            # captions = captions.transpose(0,1).unsqueeze(2)
            lengths = to_var(torch.LongTensor(lengths))            # print(captions.size())

            img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_mu, img_logv, img_z, txt_mu, txt_logv, txt_z = \
                                coupled_vae(images.detach(),captions.detach(), lengths)


            img_emb = img_z[:,:mask]
            txt_emb = txt_z[:,:mask]


            img_emb = img_emb[:,:mask]

            # current_embeddings = torch.cat( \
            #         (txt_emb.transpose(0,1).data,img_emb.unsqueeze(1).data)
            #         , 1)

            current_embeddings = np.concatenate( \
                (txt_emb.unsqueeze(0).cpu().data.numpy(),\
                 img_emb.unsqueeze(0).cpu().data.numpy())\
                ,0)

            # current_embeddings = img_emb.data
            if i:
                # result_embeddings = torch.cat( \
                result_embeddings = np.concatenate( \
                    (result_embeddings, current_embeddings) \
                    ,1)
            else:
                result_embeddings = current_embeddings

            cm_loss = cm_criterion(txt_emb, img_emb, \
                                 Variable(torch.ones(img_emb.size(0)).cuda()))

            cm_losses.update(cm_loss.data[0], img_emb.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | CM_LOSS: {cm_l:.4f}'.format(
                        batch=i,
                        size=len(loader),
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        cm_l=cm_losses.avg,
                        )
            bar.next()
        bar.finish()


        a = [((result_embeddings[0][i] - result_embeddings[1][i]) ** 2).mean() for i in range(result_embeddings.shape[0])]
        print("Validation MSE: ",np.mean(a))
        print("Validation MSE: ",np.mean(a))

        print("Computing Nearest Neighbors...")
        i = 0
        topk = []
        kss = [1,5,10]
        for k in kss:

            if i:
                print("Normalized ")
                result_embeddings[0] = result_embeddings[0]/result_embeddings[0].sum()
                result_embeddings[1] = result_embeddings[1]/result_embeddings[1].sum()

            # k = 5
            neighbors = NearestNeighbors(k, metric = 'cosine')
            neigh = neighbors
            neigh.fit(result_embeddings[1])
            kneigh = neigh.kneighbors(result_embeddings[0], return_distance=False)

            ks = set()
            for n in kneigh:
                ks.update(set(n))

            print(len(ks)/result_embeddings.shape[1])

            # a = [((result_embeddings[0][i] - result_embeddings[1][i]) ** 2).mean() for i in range(128)]
            # rs = result_embeddings.sum(2)
            # a = (((result_embeddings[0][0]- result_embeddings[1][0])**2).mean())
            # b = (((result_embeddings[0][0]- result_embeddings[0][34])**2).mean())
            topk.append(np.mean([int(i in nn) for i,nn in enumerate(kneigh)]))

        # with open(os.path.join(result_path,"retrieval.csv") , "a") as text_file:
        #     text_file.write("{}, {}, {}\n".format(tpk= 100*topk[0],
        #                               tpk2= 100*topk[1],
        #                               tpk3= 100*topk[2]))



        print("Top-{k:},{k2:},{k3:} accuracy for Image Retrieval:\n\n\t\033[95m {tpk: .3f}% \t {tpk2: .3f}% \t {tpk3: .3f}% \n".format(
                      k=kss[0],
                      k2=kss[1],
                      k3=kss[2],
                      tpk= 100*topk[0],
                      tpk2= 100*topk[1],
                      tpk3= 100*topk[2]))



def validate(coupled_vae, loader, mask, limit = 1000, metric = "cosine"):

        # cm_criterion = nn.CosineEmbeddingLoss()
        cm_criterion = nn.MSELoss()

        # VALIDATION TIME
        print('\033[92mEPOCH ::: VALIDATION ::: ' )

        # Set Evaluation Mode
        coupled_vae.eval()

        batch_time = AverageMeter()
        end = time.time()

        bar = Bar('Computing Validation Set Embeddings', max=len(loader))

        cm_losses = AverageMeter()

        for i, (images, captions, lengths) in enumerate(loader):
            if i == limit:
                break

            # Set mini-batch dataset
            images = to_var(images)
            captions = to_var(captions)

            # captions = captions.transpose(0,1).unsqueeze(2)
            lengths = to_var(torch.LongTensor(lengths))            # print(captions.size())

            img2img_out, txt2img_out, img2txt_out, txt2txt_out, img_mu, img_logv, img_z, txt_mu, txt_logv, txt_z = \
                                coupled_vae(images.detach(),captions.detach(), lengths)


            img_emb = img_z[:,:mask]
            txt_emb = txt_z[:,:mask]


            img_emb = img_emb[:,:mask]

            # current_embeddings = torch.cat( \
            #         (txt_emb.transpose(0,1).data,img_emb.unsqueeze(1).data)
            #         , 1)

            current_embeddings = np.concatenate( \
                (txt_emb.unsqueeze(0).cpu().data.numpy(),\
                 img_emb.unsqueeze(0).cpu().data.numpy())\
                ,0)

            # current_embeddings = img_emb.data
            if i:
                # result_embeddings = torch.cat( \
                result_embeddings = np.concatenate( \
                    (result_embeddings, current_embeddings) \
                    ,1)
            else:
                result_embeddings = current_embeddings

            cm_loss = cm_criterion(txt_emb, img_emb, \
                                 Variable(torch.ones(img_emb.size(0)).cuda()))

            cm_losses.update(cm_loss.data[0], img_emb.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | CM_LOSS: {cm_l:.4f}'.format(
                        batch=i,
                        size=len(loader),
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        cm_l=cm_losses.avg,
                        )
            bar.next()
        bar.finish()


        a = [((result_embeddings[0][i] - result_embeddings[1][i]) ** 2).mean() for i in range(result_embeddings.shape[0])]
        print("Validation MSE: ",np.mean(a))
        print("Validation MSE: ",np.mean(a))

        print("Computing Nearest Neighbors...")
        i = 0
        topk = []
        kss = [1,5,10]
        for k in kss:

            if i:
                print("Normalized ")
                result_embeddings[0] = result_embeddings[0]/result_embeddings[0].sum()
                result_embeddings[1] = result_embeddings[1]/result_embeddings[1].sum()

            # k = 5
            neighbors = NearestNeighbors(k, metric = 'cosine')
            neigh = neighbors
            neigh.fit(result_embeddings[1])
            kneigh = neigh.kneighbors(result_embeddings[0], return_distance=False)

            ks = set()
            for n in kneigh:
                ks.update(set(n))

            print(len(ks)/result_embeddings.shape[1])

            # a = [((result_embeddings[0][i] - result_embeddings[1][i]) ** 2).mean() for i in range(128)]
            # rs = result_embeddings.sum(2)
            # a = (((result_embeddings[0][0]- result_embeddings[1][0])**2).mean())
            # b = (((result_embeddings[0][0]- result_embeddings[0][34])**2).mean())
            topk.append(np.mean([int(i in nn) for i,nn in enumerate(kneigh)]))

        # with open(os.path.join(result_path,"retrieval.csv") , "a") as text_file:
        #     text_file.write("{}, {}, {}\n".format(tpk= 100*topk[0],
        #                               tpk2= 100*topk[1],
        #                               tpk3= 100*topk[2]))



        print("Top-{k:},{k2:},{k3:} accuracy for Image Retrieval:\n\n\t\033[95m {tpk: .3f}% \t {tpk2: .3f}% \t {tpk3: .3f}% \n".format(
                      k=kss[0],
                      k2=kss[1],
                      k3=kss[2],
                      tpk= 100*topk[0],
                      tpk2= 100*topk[1],
                      tpk3= 100*topk[2]))



