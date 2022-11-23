# texture-vq-vae-wfc
This repository contains the source code for a research project on generating "similar" textures to existing ones using a neurosymbolic method.


## Run instructions

There are two files you may want to run in this project.
The instructions for running both can be found below, but to run them successfully you must first install all required dependancies. To do this, please install everything in requirements.txt, for example by running the following command: 

`pip install -r requirements.txt`

### Training a new VQ-VAE
To train a new VQ-VAE, run the following command:

`python train_VQVAE.py --model_loc <model save location>`

This will train a new VQ-VAE and save it to `<model_loc>` in a new folder named `vqvae-%m-%d-%H-%M_LD%LD_NE%NE` where %m, %d, %H and %M represent the month, day hour and minute the training finished. The %LD represents the latent dimension of the embedding, and the %NE the number of embeddings. These can be helpful to keep track of to make sure these values match in the code for `create_similar_textures.py`

### Running the neurosymbolic generation

To run the neurosymbolc generation you must have a trained VQ-VAE either from running `train_VQVAE.py` yourself or using trained models archived in the `saved_models` folder.

To run the generation use the following command:

`python create_similar_texture.py --texture_loc <texture_loc> --vqvae_loc <vqvae_loc> --save_loc <save_loc>`

* `texture_loc` should be the path of a single image.

* `vqvae_loc` should be the path of a saved Keras model, such as the folder created by running `train_VQVAE.py`.

* `save_loc` is where you would like the resulting new texture(s) to be saved. They will be saved in .png format.
