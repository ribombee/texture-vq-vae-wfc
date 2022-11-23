# texture-vq-vae-wfc

## Running:
### Training the VQ-VAE:
To train the VQ-VAE you need to run the command:
```
python train_VQVAE.py --model_loc saved_models
```
This will train the VQ-VAE and save the new model to a folder of the format:
```
./
    saved_models/
        vqvae_%m-%d-%h-%M/
            ...
```
Where the saved model will be automatically saved within the `vqvae_%m-%d-%h-%M/` folder. e.g. `vqvae_10-29-20-11/` means 10th month, 29th day, 10th hour, and 11th minute.
