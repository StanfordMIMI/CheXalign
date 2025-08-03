import os
import argparse
import tqdm
import numpy as np
import torch
from datasets import load_from_disk, DatasetDict, Dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Get preference data from annotated completions")
    parser.add_argument("--data", type=str, default="mimic-cxr-struct-findings-generation-with-indication-completions-green-gathered", help="Data to be used.")
    parser.add_argument("--split", type=str, default="train", help="Data split to be used.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing.")
    parser.add_argument("--main_path", type=str, default="/home/dhein/Documents/CheXalign", help="Main path")
    parser.add_argument("--thresh", type=float, default=1e-8, help="Minimum value for spread between chosen and rejected")
    return parser.parse_args()

def process_img(paths):
    if isinstance(paths, str):
        paths = paths.split("|")
    return paths

def get_preference(dat, thresh, tokenizer):
    # get refs
    refs = dat["section_findings"]

    # get generated reports 
    hyps = []
    hyps.append(dat["candidate_findings_1"])
    hyps.append(dat["candidate_findings_2"])
    hyps.append(dat["candidate_findings_3"])
    hyps.append(dat["candidate_findings_4"])

    # get rewards
    rewards1 = torch.tensor(dat["green_scores1"]).unsqueeze(0)
    rewards2 = torch.tensor(dat["green_scores2"]).unsqueeze(0) 
    rewards3 = torch.tensor(dat["green_scores3"]).unsqueeze(0) 
    rewards4 = torch.tensor(dat["green_scores4"]).unsqueeze(0)

    # collect results
    rewards = torch.cat((torch.cat((rewards1,rewards2),0),torch.cat((rewards3,rewards4),0)),0)

    # get min/max
    max_val, max_idx = torch.max(rewards,0)
    min_val, min_idx = torch.min(rewards,0)
    
    # sort based on rewards 
    prompts = []
    chosen = []
    rejected = []
    reference = []
    reward_chosen = []
    reward_rejected = []
    for i in range(min_idx.size(0)):
        if abs(max_val[i].item()-min_val[i].item()) > thresh:  
            prompt = dat["prompt"][0] 
            paths = dat["image_path"][0] 
            paths = process_img(paths[:2])
            query = tokenizer.from_list_format([*[{'image': path} for path in paths], {'text': prompt}])
            reference.append(refs[i])
            prompts.append(query)
            chosen.append(hyps[max_idx[i]][i])
            rejected.append(hyps[min_idx[i]][i])
            reward_chosen.append(max_val[i].item())
            reward_rejected.append(min_val[i].item())
        else:
            pass

    if len(chosen) > 0 and len(rejected) > 0 and len(prompts) > 0:  
        return {
            "prompt": query,  
            "chosen": chosen[0],  
            "rejected": rejected[0],  
            "reference": reference[0],  
            "reward_chosen": reward_chosen[0],  
            "reward_rejected": reward_rejected[0],  
        }
    else:
        return None

def main():
    args = parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("StanfordAIMI/CheXagent-2-3b", trust_remote_code=True)

    # Load data
    dataset = load_from_disk(os.path.join(args.main_path,"data",args.data))
    dataset = dataset[args.split]

    # Main loop    
    reward_chosen = []
    reward_rejected = []
    local_candidates = []
    for i in tqdm.tqdm(dataset.iter(batch_size=args.batch_size), total=len(dataset) // args.batch_size):
        obj = get_preference(i, args.thresh, tokenizer)
        if obj:
            local_candidates.append(obj)
            reward_chosen += [obj["reward_chosen"]]
            reward_rejected += [obj["reward_rejected"]]

    print(f"Reward chosen mean: {np.mean(reward_chosen)}")
    print(f"Reward chosen std: {np.std(reward_chosen)}")
    print(f"Reward rejected mean: {np.mean(reward_rejected)}")
    print(f"Reward rejected std: {np.std(reward_rejected)}")
    if args.split == "val":
        dataset_new = DatasetDict({"validation": Dataset.from_list(local_candidates)})
    else:
        dataset_new = DatasetDict({args.split: Dataset.from_list(local_candidates)})

    print(dataset_new)
    print(dataset_new["train"][0])

    output_dir = os.path.join(args.main_path,"data",args.data+"-final")
    os.makedirs(output_dir, exist_ok=True)
    dataset_new.save_to_disk(output_dir)

if __name__ == '__main__':
    main()

