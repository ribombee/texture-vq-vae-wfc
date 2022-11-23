# texture-vq-vae-wfc
This repository contains the source code for a research project on generating "similar" textures to existing ones using a neurosymbolic method.


## Run instructions

There are two files you may want to run in this project.
The instructions for running both can be found below, but to run them successfully you must first install all required dependancies. To do this, please install everything in requirements.txt, for example by running the following command: 

`pip install -r requirements.txt`

### Training a new VQ-VAE
To train a new VQ-VAE, run the following command:

`python train_VQVAE.py --model_loc <model save location>`

This will train a new VQ-VAE and save it to `<model_loc>` in a new folder named `vqvae_%m_%d_%H_%M` where %m, %d, %H and %M represent the month, day hour and minute the training finished.

### Running the neurosymbolic generation

To run the neurosymbolc generation you must have a trained VQ-VAE either from running `train_VQVAE.py` yourself or using trained models archived in the `saved_models` folder.

To run the generation use the following command:

`python create_similar_texture.py --texture_loc <texture_loc> --vqvae_loc <vqvae_loc> --save_loc <save_loc>`

* `texture_loc` should be the path of a single image.

* `vqvae_loc` should be the path of a saved Keras model, such as the folder created by running `train_VQVAE.py`.

* `save_loc` is where you would like the resulting new texture(s) to be saved. They will be saved in .png format.
