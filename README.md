# Integrating Video and Language with Generative Adversarial Networks

![](abstract/media/mmmGanwide.png)


__TO DO:__

[Coupled Auto-Encoder Network](https://github.com/vglsd/thesis/blob/master/abstract/m3GAN-entavelis.pdf):

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