import os 
import argparse
import pandas as pd
from PIL import Image

''' Making a cropper script with arg parse is good because then you can parallelize it in the commandline
like run the script in different jobs from the command line rather than implementing a threading class here. 

'''
def get_args():
    parser = argparse.ArgumentParser(description="Extract image patches with stride.")
    parser.add_argument("--file", type=str, required=True, help="Path to the input file")
    parser.add_argument("--patch_size", type=int, required=True, help="Patch size (e.g., 64 for 64x64)")
    parser.add_argument("--stride", type=int, required=True, help="Stride length for patches")
    parser.add_argument("--odir", type=str, required=True, help="path to output directory")
    return parser.parse_args()

def extract_and_save_patches(file, patch_size, stride, odir="odir", csv_name="patches.csv"):
    if not os.path.exists(odir):
        os.makedirs(odir)

    img = Image.open(file).convert("RGB")
    width, height = img.size
    basename = os.path.splitext(os.path.basename(file))[0]
    saved_paths = []
    ji = 0
    
    for row, y in enumerate(range(0, height - patch_size + 1, stride)):
        for col, x in enumerate(range(0, width - patch_size + 1, stride)):
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            fname = f"{basename}_r{row}_c{col}.png"
            fpath = os.path.join(odir, fname)
            patch.save(fpath)
            saved_paths.append(fpath)
            

    # save file paths to CSV with pandas
    df = pd.DataFrame(saved_paths, columns=["patch_path"])
    df.to_csv(os.path.join(odir, csv_name), index=False)
    return saved_paths

if __name__ == "__main__":
    args = get_args()
    print(f"File: {args.file} | Patch size: {args.patch_size} | Stride: {args.stride}")
    
    paths = extract_and_save_patches(args.file, patch_size=args.patch_size, 
                                    stride=args.stride, odir = args.odir)
    print(f"Saved {len(paths)} patches, file list written to patches.csv")
