# Integrating Video and Language with Generative Adversarial Networks

![](abstract/media/mmmGanwide.png)

[Architecture Description](https://github.com/vglsd/thesis/blob/master/abstract/m3GAN-entavelis.pdf)

__TO DO:__

[Coupled Auto-Encoder Network](https://github.com/vglsd/thesis/blob/master/abstract/m3GAN-entavelis.pdf):
- Do the _Karpathy Split_
- Negative Loss (j is a random image's index from the batch):

    ![equation](http://latex.codecogs.com/gif.latex?Loss_%7Bcm%7D%28i%29%29%20%3D%20max%28-0.001%2C%20mse%28E_%7Btxt%7D%28i%29%2CE_%7Bimg%7D%28i%29%29%20-%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5Ek%7Bmse%28E_%7Btxt%7D%28i%29%2CE_%7Bimg%7D%28j_k%29%29%7D%7D%7Bk%7D%29)
    
    
    
- Try masking common embedding: 
    - degree of masking
- Try different embedding errors: 
    - CosineEmbLoss
    - HingeEmbLoss
- Try propagate errors differently:
    - Chain Encoder-Decoder
    - Propagate only reconstruction error at decoders
- Use 256x256 image sizes instead of 128x128
- Randomly crop images for normalization
- Train first one and then the other
- Train only the encoders on some steps
- Try Variational AE:
    - how wide range -> sigma
    

![](abstract/media/pretraining.png)


### Based on:

- [DiscoGAN](https://github.com/SKTBrain/DiscoGAN)
- [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
- [PyTorch-Seq2Seq Example](https://github.com/howardyclo/pytorch-seq2seq-example/)
- [CNN-Sentence_Classification](https://github.com/A-Jacobson/CNN_Sentence_Classification)
- [PyTorch Tutorial: Image Captioning](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)
- [Pytorch-classification](https://github.com/bearpaw/pytorch-classification)

---------------

Promotor: Professor Marie-Francine Moens

Daily Supervisor: Guillem Collell Talleda