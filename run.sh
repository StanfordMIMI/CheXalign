main_path=/home/dhein/Documents/CheXalign
betas=(0.1 0.05 0.01)
annotators=(green)
gradient_accumulation_steps=4
batch_size=4
algorithm=dpo
for beta in "${betas[@]}"; do
    for annotator in "${annotators[@]}"; do
        ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ./scripts/configs/deepspeed_zero3.yaml \
            ./scripts/run_${algorithm}.py ./scripts/configs/config_chexagent_${algorithm}.yaml --beta=${beta} \
            --model_name_or_path=StanfordAIMI/CheXagent-2-3b --max_length=3072 --preprocessing_num_workers=32 \
            --gradient_accumulation_steps=${gradient_accumulation_steps} --per_device_train_batch_size=${batch_size}\
            --dataset_mixer=mimic-cxr-struct-findings-generation-with-indication-completions-${annotator}-gathered-final \
            --output_dir=${main_path}/results/chexagent-2-3b-${algorithm}-${beta}-${alpha}-${batch_size}-${gradient_accumulation_steps}-${annotator} \
            --save_steps=2000 --save_total_limit=20
    done
done 