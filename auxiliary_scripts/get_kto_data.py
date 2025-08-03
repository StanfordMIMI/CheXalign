import argparse
import os
import tqdm
from datasets import DatasetDict, load_from_disk, Dataset

def process(batch):
    records = []
    for prompt, chosen, rejected in zip(batch["prompt"],
                                         batch["chosen"],
                                         batch["rejected"]):
        records.append({"prompt": prompt,
                        "completion": chosen,
                        "label": True})
        records.append({"prompt": prompt,
                        "completion": rejected,
                        "label": False})
    return records

def main(dataset_id,main_path):
    splits = ["train"]
    output_dir = os.path.join(main_path,"data/",dataset_id + "_kto")
    batch_size = 1

    dataset = load_from_disk(os.path.join(main_path,"data/",dataset_id))
    print(dataset["train"])

    # Main loop
    for split in splits:
        local_candidates = []
        for i in tqdm.tqdm(
            dataset[split].iter(batch_size=batch_size),
            total=len(dataset) // batch_size,
            ):
            obj = process(i)
            if isinstance(obj, dict):
                obj = [obj]
            if obj is None:
                continue
            local_candidates.extend(obj)

        dataset_new = DatasetDict({split: Dataset.from_list(local_candidates)})
        print(dataset_new["train"])
        print(dataset_new["train"][0])
        print(dataset_new["train"][1])
        print("Saving the dataset")
        os.makedirs(output_dir, exist_ok=True)
        dataset_new.save_to_disk(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some datasets.")
    parser.add_argument("--dataset_id", type=str, required=True, help="The ID of the dataset to load")
    parser.add_argument("--main_path", type=str, default="/home/dhein/Documents/CheXalign", help="Main_path")
    args = parser.parse_args()
    main(args.dataset_id,args.main_path)
