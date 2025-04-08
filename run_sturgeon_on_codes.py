import csv
import time
from tqdm import tqdm
import subprocess
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_command(command, timeout=None):
    start_time = time.time()
    while True:
        try:
            subprocess.run(command, shell=True, timeout=timeout, check=True)
            end_time = time.time()
            return end_time - start_time
        except subprocess.CalledProcessError:
            print(f"Command failed: {command}, retrying...")
        except subprocess.TimeoutExpired:
            print(f"Command timed out: {command}, retrying...")

def log_to_csv(log_file, data):
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def generate_text_file(out_path, idx):
    text_command = f"python scheme2output.py --schemefile {f'{out_path}/schema.schema'} --outfile {f'{out_path}/new_{idx}.txt'} --pattern-hard --count-soft --size 16 16 --randomize {idx}"
    total_time = 0
    success = False
    for _ in range(2):  # Retry once if it fails
        try:
            start_time = time.time()
            subprocess.run(text_command, shell=True, timeout=600)
            end_time = time.time()
            total_time += end_time - start_time
            success = True
            break
        except subprocess.TimeoutExpired:
            print(f"Timeout expired for {out_path} new_{idx}.txt, retrying...")
            total_time += 600
    return total_time if success else None

def run_sturgeon_commands(code_path, log_file='command_log.csv'):
    all_code_paths = Path(code_path).rglob("*.txt")

    for img_code_path in tqdm(all_code_paths):
        img_name = str(img_code_path.stem)
        img_code_path = str(img_code_path)
        out_path = str(code_path) + "/" + img_name

        # Create "tile" file
        tile_command = f"python input2tile.py --textfile {img_code_path} --outfile {f'{out_path}/tile.tile'}"
        tile_time = run_command(tile_command)
        if not os.path.exists(f'{out_path}/tile.tile'):
            print(f"Failed to generate tile file for {img_name}")
            log_to_csv(log_file, [img_name, tile_time, None, None, None, 'Failed'])
            continue

        # Create "scheme" file
        schema_command = f"python tile2scheme.py --tilefile {out_path}/tile.tile --outfile {f'{out_path}/schema.schema --pattern nbr-plus --count-divs 1 1'}"
        schema_time = run_command(schema_command)
        if not os.path.exists(f'{out_path}/schema.schema'):
            print(f"Failed to generate schema file for {img_name}")
            log_to_csv(log_file, [img_name, tile_time, schema_time, None, None, 'Failed'])
            continue

        # Generate new text files using threading
        total_times = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(generate_text_file, out_path, idx) for idx in range(5)]
            for future in as_completed(futures):
                total_times.append(future.result())

        if all(total_times):
            log_to_csv(log_file, [img_name, tile_time, schema_time] + total_times + ['Success'])
        else:
            log_to_csv(log_file, [img_name, tile_time, schema_time] + total_times + ['Failed'])

def __parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run Sturgeon on codes")
    parser.add_argument("code_path", type=str, help="Path to the directory containing the codes")
    return parser.parse_args()

if __name__ == "__main__":
    code_path = __parse_args().code_path
    run_sturgeon_commands(code_path)