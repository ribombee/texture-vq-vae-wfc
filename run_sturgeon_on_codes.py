from tqdm import tqdm
import subprocess
from pathlib import Path
import os
import time

def run_command(command, timeout=None):
    while True:
        try:
            subprocess.run(command, shell=True, timeout=timeout, check=True)
            return True
        except subprocess.CalledProcessError:
            print(f"Command failed: {command}, retrying...")
        except subprocess.TimeoutExpired:
            print(f"Command timed out: {command}, retrying...")

def run_sturgeon_commands(code_path):
    all_code_paths = Path(code_path).rglob("*.txt")

    for img_code_path in tqdm(all_code_paths):
        img_name = str(img_code_path.stem)
        img_code_path = str(img_code_path)
        out_path = str(code_path) + "/" + img_name

        # Create "tile" file
        tile_command = f"python input2tile.py --textfile {img_code_path} --outfile {f'{out_path}/tile.tile'}"
        run_command(tile_command)
        if not os.path.exists(f'{out_path}/tile.tile'):
            print(f"Failed to generate tile file for {img_name}")
            continue

        # Create "scheme" file
        schema_command = f"python tile2scheme.py --tilefile {out_path}/tile.tile --outfile {f'{out_path}/schema.schema --pattern nbr-plus --count-divs 1 1'}"
        run_command(schema_command)
        if not os.path.exists(f'{out_path}/schema.schema'):
            print(f"Failed to generate schema file for {img_name}")
            continue

        # Generate new text files
        for idx in range(5):
            text_command = f"python scheme2output.py --schemefile {f'{out_path}/schema.schema'} --outfile {f'{out_path}/new_{idx}.txt'} --pattern-hard --count-soft --size 16 16 --randomize {idx}"
            try:
                subprocess.run(text_command, shell=True, timeout=600)
            except subprocess.TimeoutExpired:
                print(f"Timeout expired for {img_name} new_{idx}.txt, retrying...")
                try:
                    subprocess.run(text_command, shell=True, timeout=600)
                except subprocess.TimeoutExpired:
                    print(f"Failed to generate new_{idx}.txt for {img_name} after retrying")
                    continue

def __parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run Sturgeon on codes")
    parser.add_argument("code_path", type=str, help="Path to the directory containing the codes")
    return parser.parse_args()

if __name__ == "__main__":
    code_path = __parse_args().code_path
    run_sturgeon_commands(code_path)