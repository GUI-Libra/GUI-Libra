dir='2026_2_3'
add_template=1
reasoning=1
blocks_dir=path/to/your/blocks/dir # the images folder from GUI-Libra/Offline-Evaluation/Multimodal-Mind2Web_blocks_images/release_images


model_paths=(
path/to/your/model
)
 
 

exp_names=(
exp_name
)


ports=(8000 8001 8002 8003 8004 8005 8006 8007)


for split in  'domain'  'website'   'task'  
do
    temperature=0.0
    for id in 0
    do
        for i in "${!model_paths[@]}"; do
            model_path="${model_paths[$i]}"
            exp_name="${exp_names[$i]}"
            port="${ports[$i]}"
            echo "Processing split: ${split}, id: ${id}, model_path: ${model_path}"
            python guilibra_plan.py --model ${model_path}  --input_file data/samples/cross_${split}_blocks_natural.jsonl \
            --output_file data/${dir}/${exp_name}/cross_${split}/plan_${id}_temperature${temperature}.jsonl \
            --blocks ${blocks_dir}/cross_${split}  \
            --temperature ${temperature} \
            --port ${port} \
            --add_template ${add_template} --reasoning ${reasoning} &
        done
    done
    wait


    temperature=1.0
    for id in 0 1 2 3 
    do
        for i in "${!model_paths[@]}"; do
            model_path="${model_paths[$i]}"
            exp_name="${exp_names[$i]}"
            port="${ports[$i]}"
            echo "Processing split: ${split}, id: ${id}, model_path: ${model_path}"
            python guilibra_plan.py --model ${model_path}  --input_file data/samples/cross_${split}_blocks_natural.jsonl \
            --output_file data/${dir}/${exp_name}/cross_${split}/plan_${id}_temperature${temperature}.jsonl \
            --blocks ${blocks_dir}/cross_${split}  \
            --temperature ${temperature} \
            --port ${port} \
            --add_template ${add_template} --reasoning ${reasoning} &
        done
        wait
    done
done




