import os
import argparse
from datasets import load_from_disk, DatasetDict, concatenate_datasets

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_path",    type=str, default="/home/dhein/Documents/CheXalign", help="Main data directory")
    parser.add_argument("--data",         type=str, default="mimic-cxr-struct-findings-generation-with-indication-completions", help="Base dataset name")
    parser.add_argument("--split",        type=str, default="train", help="Dataset split to gather")
    parser.add_argument("--judge",        type=str, default="green", help="Scoring method prefix")
    parser.add_argument("--n_completions",type=int, default=4, help="Number of completions per sample")
    parser.add_argument("--chunk_size",   type=int, default=1000, help="Number of samples per chunk")
    parser.add_argument("--start",        type=int, default=0, help="Start index")
    parser.add_argument("--end",          type=int, default=1000, help="End index")
    return parser.parse_args()

def main():
    args = parse_args()

    all_datasets = []
    for i in range(args.start, args.end, args.chunk_size):
        chunk_end = min(i + args.chunk_size, args.end)
        print(f"Processing chunk from {i} to {chunk_end}")

        # Load the first completion’s chunk
        base_path = os.path.join(
            args.main_path,
            "data/chunks",
            f"{args.data}-{args.judge}-1-chunk_{i}_{chunk_end}"
        )
        ds = load_from_disk(base_path)[args.split]

        # Append the other completions’ scores
        for completion in range(2, args.n_completions + 1):
            temp_path = os.path.join(
                args.main_path,
                "data/chunks",
                f"{args.data}-{args.judge}-{completion}-chunk_{i}_{chunk_end}"
            )
            temp_ds = load_from_disk(temp_path)[args.split]
            ds = ds.add_column(f"{args.judge}_scores{completion}", temp_ds[f"{args.judge}_scores{completion}"])

        all_datasets.append(ds)

    # Concatenate and wrap in a DatasetDict
    final_ds = concatenate_datasets(all_datasets)
    final_dataset = DatasetDict({args.split: final_ds})
    print(final_dataset)

    # Save
    output_path = os.path.join(args.main_path, f"data/{args.data}-{args.judge}-gathered")
    final_dataset.save_to_disk(output_path)
    print(f"\nSaved updated dataset to: {output_path}")

if __name__ == "__main__":
    main()
