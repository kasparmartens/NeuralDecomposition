# NeuralDecomposition

This is a PyTorch implementation of our AISTATS paper [Neural Decomposition: Functional ANOVA with Variational Autoencoders](arxiv.org/abs/2006.14293) (Märtens & Yau, 2020)

### Implementation 

* (`decoder.py`)[https://github.com/kasparmartens/NeuralDecomposition/blob/master/ND/decoder.py] impements the class for the decomposable CVAE decoder, for the special case of a one-dimensional latent variable *z* and one-dimensional covariate *c*.
* (encoder.py)[ND/encoder.py] implements the class for a standard CVAE encoder
* (CVAE.py)[ND/CVAE.py] provides a wrapper to combine the decoder and encoder for training purposes

An example colab notebook will be added here shortly. 
