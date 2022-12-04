import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
from WFC_train import extract_patterns, get_unique_patterns
from create_similar_texture import load_model, load_texture
import pickle

import threading

def train_texture_wfc(texture_codes) -> tuple[int,int,int]:
    wrapping = False
    pattern_height = 3
    pattern_width = 3
    print(pattern_height)
    row_offset = 1
    col_offset = 1

    all_patterns = extract_patterns(texture_codes, pattern_height, pattern_width,
                                    row_offset=row_offset, col_offset=col_offset,
                                    wrapping=wrapping)

    unique_patterns = get_unique_patterns(all_patterns)

    return len(unique_patterns)


class TextureThread(threading.Thread):
    def __init__(self, threadId:int, num_patterns_per_dim:dict, lock:threading.Lock, model_tuple:tuple, encoder, quantizer, base_texture_path:str, texture_paths:list, start_texture_index:int, end_texture_index:int):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.num_patterns_per_dim = num_patterns_per_dim
        self.lock = lock
        self.model_tuple = model_tuple
        self.encoder = encoder
        self.quantizer = quantizer
        self.base_texture_path = base_texture_path
        self.texture_paths = texture_paths
        self.start_texture_index = start_texture_index
        self.end_texture_index = end_texture_index
    
    def run(self):
        print(f"STARTING THREAD WITH ID={self.threadId} over range {self.start_texture_index}:{self.end_texture_index} ...")
        values = []
        our_textures = self.texture_paths[self.start_texture_index:self.end_texture_index]
        for texture_path in tqdm(our_textures):

            _, normalized_texture = load_texture(f".\\{self.base_texture_path}\\{texture_path}")

            encoded_texture = self.encoder.predict(np.expand_dims(normalized_texture, 0)) # Expanding dims to have batch size as well
            flat_encoded_texture = encoded_texture.reshape(-1, encoded_texture.shape[-1])
            texture_codes = self.quantizer.get_code_indices(flat_encoded_texture)
            texture_codes = texture_codes.numpy().reshape(encoded_texture.shape[:-1])
            
            n_unique_patterns = train_texture_wfc(texture_codes)
            values.append(n_unique_patterns)
        
        self.lock.acquire()
        self.num_patterns_per_dim[self.model_tuple] += values
        self.lock.release()
        print(f"FINISHED THREAD WITH ID={self.threadId} !")


if __name__ == '__main__':
    num_patterns_per_dim = {}

    BASE_VQVAE_PATH = '.\\all_vqvae_models.\\neurosymbolic_models'
    BASE_TEXTURE_PATH = 'validation_data'

    vqvae_folders = os.listdir(BASE_VQVAE_PATH)
    texture_paths = os.listdir('.\\' + BASE_TEXTURE_PATH)

    print("BEGINNING ITERATION OVER MODELS")
    
    # print(texture_paths[:10])

    for model_dimension_folder in tqdm(vqvae_folders):
        split_folder = model_dimension_folder.rstrip().split('-')
        LATENT_DIM = int(split_folder[0])
        NUM_EMBEDDINGS = int(split_folder[1])
        
        # We've already done this:
        # if (LATENT_DIM,NUM_EMBEDDINGS) not in {(64,16),(256,8),(256,16)}:
        #     continue
        
        print(f"COMPUTING ON MODEL {LATENT_DIM}x{NUM_EMBEDDINGS} ...")
        
        inner_folder_list = os.listdir(f"{BASE_VQVAE_PATH}/{model_dimension_folder}")
        vqvae_path = ""
        
        if len(inner_folder_list) == 1:
            # There is only one folder inside the LD-NE folder, that is the model folder
            vqvae_path = f"{BASE_VQVAE_PATH}\\{model_dimension_folder}\\{inner_folder_list[0]}"
        else:
            # The LD-NE folder is the model folder
            vqvae_path = f"{BASE_VQVAE_PATH}\\{model_dimension_folder}"
            
        vqvae_path = Path(vqvae_path)
        
        model = load_model(vqvae_path)
        
        num_patterns_per_dim[(LATENT_DIM,NUM_EMBEDDINGS)] = []
        
        threadLock = threading.Lock()
        threads = []
        
        encoder = model.get_layer("encoder")
        quantizer = model.get_layer("vector_quantizer")
        
        thread1 = TextureThread(0, num_patterns_per_dim, threadLock, (LATENT_DIM,NUM_EMBEDDINGS), encoder, quantizer,  BASE_TEXTURE_PATH, texture_paths, 0, 470)
        thread2 = TextureThread(1, num_patterns_per_dim, threadLock, (LATENT_DIM,NUM_EMBEDDINGS), encoder, quantizer, BASE_TEXTURE_PATH, texture_paths, 470*1, 470*2)
        thread3 = TextureThread(2, num_patterns_per_dim, threadLock, (LATENT_DIM,NUM_EMBEDDINGS), encoder, quantizer, BASE_TEXTURE_PATH, texture_paths, 470*2, 470*3)
        thread4 = TextureThread(3, num_patterns_per_dim, threadLock, (LATENT_DIM,NUM_EMBEDDINGS), encoder, quantizer, BASE_TEXTURE_PATH, texture_paths, 470*3, 470*4)
        
        thread1.start(); thread2.start(); thread3.start(); thread4.start()
        
        threads = [thread1, thread2, thread3, thread4]
        
        for thread in threads:
            thread.join()
        
        print(f"DONE LD={LATENT_DIM}, NE={NUM_EMBEDDINGS}")
        
        pickle.dump(num_patterns_per_dim, open("num_patterns_per_dim_v3.pickle", "wb"))

    print("DONE")