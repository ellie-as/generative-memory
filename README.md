
### A Generative Model of Memory Construction and Consolidation

Paper: https://www.nature.com/articles/s41562-023-01799-z 

Code for modelling consolidation as teacher-student learning, in which initial representations of memories are replayed to train a generative model. This repo contains four notebooks to reproduce the results in the paper.

The modern Hopfield network code is based on https://github.com/ml-jku/hopfield-layers, which accompanies Ramsauer et al. (2020) (https://arxiv.org/abs/2008.02217).

#### Installation

Each notebook installs the dependencies required with pip. This code was tested on MacOS with Python 3.10.9.

(Note that the previous Colab link has been removed, as Colab no longer supports the version of Python required to exactly reproduce the results.)

#### 1. Consolidation simulation.ipynb

For the Shapes3D dataset, the notebook:

* Encodes 10000 images (i.e. memories) in the modern Hopfield network.
* Gives the Hopfield network 10000 random noise inputs, and gets the outputs (which should be the stored memories). The outputs of this stage are saved in the mhn_memories folder for re-use; to regenerate from scratch delete the contents of the folder.
* Trains a variational autoencoder on the 'memories', and tests its recall.
* Generates plots of semantic decoding accuracy and reconstruction error over the course of training.
* Displays the latent space (projected into 2D and colour coded by class).
* Runs a set of other tests, e.g. of interpolation between items and vector arithmetic.
* Saves the outputs to a PDF in the 'outputs' folder, together with the loss history.

If model weights for the dataset are present in the model_weights folder, these will be used to generate the results (except for the plot of semantic decoding accuracy and reconstruction error over the course of training). Move the model weights to train from scratch. 

It then implements the extended model:
* Using the VAE trained in the previous step, a new modern Hopfield network encodes memories with both a sensory (poorly predicted) and conceptual (predictable) component, where the latter is simply the VAE's latent variables.
* Examples of recall in the extended version of the model are saved in the 'hybrid_model' folder.
* The simulation is run at a higher and lower error threshold for encoding.

#### 2. Exploring prototypical distortions.ipynb

This notebook contains code to demonstrate that generative networks such as VAEs make their outputs more prototypical, demonstrated using the MNIST dataset. 

#### 3. Memory distortions in the extended model.ipynb

This notebook explores memory distortions in the extended model (using the model trained in '1. Consolidation simulation.ipynb'):

* A plot of error and number of features stored against error threshold for encoding is generated.
* The Carmichael simulation is run. To model an external conceptual context being encoded, the original image is stored in the modern Hopfield network along with activation of a given concept, represented as the latent variables for that class. 

#### 4. Boundary extension and contraction.ipynb

This notebook contains code for exploring boundary extension and contraction in a VAE trained on the Shapes3D dataset. We test boundary extension and contraction in the basic model by giving it a range of artificially 'zoomed in’ or 'zoomed out’ images, adapted from Shapes3D scenes not seen during training, and observing the outputs. The model trained in '1. Consolidation simulation.ipynb' is used. 

#### 5. Imagination continued.ipynb

This notebook contains code for generating examples of particular classes with Monte Carlo sampling.

#### 4. Few-shot learning testing.ipynb

This notebook compares few-shot learning with the latent representations as features to few-shot learning with alternative representations.

