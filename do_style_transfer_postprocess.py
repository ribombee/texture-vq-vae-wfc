from tqdm import tqdm



def run_style_transfer():
    # TODO
    pass


def run_on_all_outputs(input_dir, output_dir):
    # TODO: make it do the thing
    pass


 if __name__ == "__main__":
    import argparse
    import os
    from PIL import Image

    parser = argparse.ArgumentParser(description="Post-process style transfer results.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the style transfer results.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the post-processed images.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(args.input_dir)):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)

            # Open the image and convert it to RGB
            image = Image.open(input_path).convert("RGB")

            # Save the image in a standard format (e.g., PNG)
            image.save(output_path)

            print(f"Processed {input_path} and saved to {output_path}")