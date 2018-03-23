# Integrating Video and Language with Generative Adversarial Networks

I am going to use the README as a tracker of Goals per Week,

__Goals for Friday 23/3:__

Having a trainable Model of Seq2Seq AE and Im2Im AE

_Subgoals:_
- ~~Train S2S with GloVe Embeddings~~ __Done!__
- Train I2I


### Questions

- How to create [Sentence Embeddings](http://forum.opennmt.net/t/sentence-embeddings-for-english/1389) with [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)

    - __Answer:__ Wait for the bug to be fixed...
    
- Is the size of the RNN Layer the siez of the latent dimension?
### Notes

- Empty lines cause translate to crash
- Check the loss functions used in [Show, Adapt and Tell](https://github.com/tsenghungchen/show-adapt-and-tell#mscoco-captioning-dataset)

### Based on:

- [DiscoGAN](https://github.com/SKTBrain/DiscoGAN)
- [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
