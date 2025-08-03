import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_from_disk, DatasetDict, Dataset
from evaluation_chexbench.models import CheXagent

def main(model_name="CheXagent", num_beams=1):
    """Findings Generation for CheXagent supporting multi-gpu inference"""
    # constants
    data_path = "data/mimic-cxr-struct-findings-generation-with-indication"
    save_dir = "data/mimic-cxr-struct-findings-generation-with-indication-completions"
    n_draws = 4
    split = "train"

    # load dataset 
    dataset = load_from_disk(data_path)
    dataset = dataset[split]
    accelerator = Accelerator()

    # load the model
    model = CheXagent(device=f"cuda:{accelerator.process_index}")
    accelerator.wait_for_everyone()

    # inference
    results = []
    for sample_idx, sample in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        if sample_idx % accelerator.num_processes != accelerator.process_index:
            continue
        
        candidates = {}
        for i in range(1, n_draws + 1):
            candidates[f'candidate_findings_{i}'] = model.generate(
                sample["image_path"][:2],
                f'Given the indication: "{sample["text"]}", write a structured findings section for the CXR.',
                num_beams=num_beams,
                do_sample=True  
            )

        result = {
            "data_source": sample["data_source"],
            "image_path": sample["image_path"],
            "section_indication": sample["text"],
            "section_findings": sample["target_text"],
            "prompt": f'Given the indication: "{sample["text"]}", write a structured findings section for the CXR.'
        }
        
        result.update(candidates)
        results.append(result)

    # gather results from multiple processes
    results = gather_object([results])
    
    if accelerator.is_main_process:
        flattened = [item for sublist in results for item in sublist]
        ds = Dataset.from_list(flattened)
        DatasetDict({'train': ds}).save_to_disk(save_dir)

if __name__ == '__main__':
    main()