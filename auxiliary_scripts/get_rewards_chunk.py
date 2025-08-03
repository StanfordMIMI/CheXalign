import os
import re
import argparse
from datasets import load_from_disk, DatasetDict
from GREEN.green_score import GREEN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="mimic-cxr-struct-findings-generation-with-indication-completions", help="Data to be used.")
    parser.add_argument("--split", type=str, default="train", help="Data split to be used.")
    parser.add_argument("--completion", type=int, default=1, help="")
    parser.add_argument("--main_path", type=str, default="/home/dhein/Documents/CheXalign", help="Main path")
    parser.add_argument("--start_idx", type=int, required=True, help="Start index for processing chunk")
    parser.add_argument("--end_idx", type=int, required=True, help="End index for processing chunk")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # set up GREEN
    model_name = "StanfordAIMI/GREEN-radllama2-7b"
    green_scorer = GREEN(model_name, output_dir=".")
    
    # Load data
    dataset = load_from_disk(os.path.join(args.main_path, "data", args.data))
    print(f"\nOriginal dataset: {dataset}")
    
    # Select chunk
    dataset = dataset[args.split].select(range(args.start_idx, args.end_idx))
    print(dataset)
    
    clean_text = lambda x: re.sub(r"\s+", " ", re.sub(r"\[.*?\]", "", x).replace("**", "")).strip().lower()
    hyps = [clean_text(sample[f"candidate_findings_{args.completion}"]) for sample in dataset]
    refs = [clean_text(sample["section_findings"]) for sample in dataset]
    
    # Get GREENs
    mean, std, green_score_list, summary, results_df = green_scorer(refs, hyps)
    
    # Save format and save data
    dataset = DatasetDict({args.split: dataset.add_column(f"green_scores{args.completion}", green_score_list)})
    print(f"\nDataset with GREEN scores: {dataset}")
    
    # Modified output path to include chunk information
    output_path = os.path.join(
        args.main_path,
        "data/chunks",
        f"{args.data}-green-{args.completion}-chunk_{args.start_idx}_{args.end_idx}"
    )
    dataset.save_to_disk(output_path)
    print(f"\nSaved updated dataset to: {output_path}")

if __name__ == '__main__':
    main()
