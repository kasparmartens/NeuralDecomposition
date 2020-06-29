# NeuralDecomposition

This is a PyTorch implementation of our AISTATS paper [Neural Decomposition: Functional ANOVA with Variational Autoencoders](arxiv.org/abs/2006.14293) (Märtens & Yau, 2020)

![](fig/feature_level_decomposition.png)

![](fig/ND_schema.png)

### Implementation 

* [`decoder.py`](ND/decoder.py) impements the class for the decomposable CVAE decoder, for the special case of a one-dimensional latent variable *z* and one-dimensional covariate *c*.
* [`encoder.py`](ND/encoder.py) implements the class for a standard CVAE encoder
* [`CVAE.py`](ND/CVAE.py) provides a wrapper to combine the decoder and encoder for training purposes

### Example notebook

See the  for an example on synthetic data:

* [open in GitHub](toy_example.ipynb)
* [open in Colab](https://colab.research.google.com/github/kasparmartens/NeuralDecomposition/blob/master/toy_example.ipynb)
