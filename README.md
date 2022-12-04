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

To run the neurosymbolic generation you must have a trained VQ-VAE either from running `train_VQVAE.py` yourself or using trained models archived in the `saved_models` folder.

To run the generation use the following command:

`python create_similar_texture.py --texture_loc <texture_loc> --vqvae_loc <vqvae_loc> --save_loc <save_loc>`

* `texture_loc` should be the path of a single image.

* `vqvae_loc` should be the path of a saved Keras model, such as the folder created by running `train_VQVAE.py`.

* `save_loc` is where you would like the resulting new texture(s) to be saved. They will be saved in .png format.

### Running the neurosymbolic average pattern count code
There are two Python files that run and output the data:  
- `count_num_patterns.py` will count the number of patterns per texture, per model. So in essence, for every single model in `.\all_vqvae_models\neurosymbolic_models` it will calculate the number of patterns for each texture in `.\validation_data`. This is a multithreaded program and will run on 4 threads. Each model on my i5-8600k took about 10-15 minutes each. For 24 models total it took me 4-6 hours of runtime.  
To run the models you simply type: `python count_num_patterns.py`  
This will save the data under `num_patterns_per_dim_v3.pickle`. You can change where it saves it to. It's important to remember this for the next step that computes the averages.  
If you only want to run certain models, edit Lines 82 and 83.
- `pattern_counter.py` will simply compute the average number of patterns for each model. The first commented part of the program will combine any pickle files you have into one. Yes this is bad style for now. The second part will take whatever pickle file your data is in and output a nicely formatted table of average values.  
You can run this program by typing: `python pattern_counter.py`