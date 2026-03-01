#!/bin/bash

export OPENAI_API_KEY="your_openai_api_key"

model_name='gpt-4o'
blocks_dir='path/to/blocks_images'

for split in 'domain' 'website' 'task'
do
    echo "Processing split: ${split}"
    python gpt_plan.py \
        --gpt_model ${model_name} \
        --input_file data/samples/cross_${split}_blocks.jsonl \
        --output_file data/${model_name}_results/cross_${split}_plan.jsonl \
        --blocks ${blocks_dir}/cross_${split}
    echo "Completed split: ${split}"
done
