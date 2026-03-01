dir='2026_1_24'
add_template=1
reasoning=1
sem_limit=64
blocks_dir='path/to/blocks_images'

model_paths=(
path/to/GUI-R1-7B
path/to/GUI-R1-3B
)



exp_names=(
GUI-R1_7B_baseline_removepress_v2_natural
GUI-R1_3B_baseline_removepress_v2_natural
)

agent_family='gui_r1'

ports=(8006 8007)


for split in   'domain' 'website'  'task' 
do
    temperature=0.0
    for id in 0
    do
        for i in "${!model_paths[@]}"; do
            model_path="${model_paths[$i]}"
            exp_name="${exp_names[$i]}"
            port="${ports[$i]}"
            echo "Processing split: ${split}, id: ${id}, model_path: ${model_path}"
            python uir1_plan.py --model ${model_path}  --input_file data/samples/cross_${split}_blocks_natural.jsonl \
            --output_file data/${dir}/${exp_name}/cross_${split}/plan_${id}_temperature${temperature}.jsonl \
            --blocks ${blocks_dir}/cross_${split}  \
            --temperature ${temperature} \
            --port ${port} \
            --sem_limit ${sem_limit} --agent_family ${agent_family}  &
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
            python uir1_plan.py --model ${model_path}  --input_file data/samples/cross_${split}_blocks_natural.jsonl \
            --output_file data/${dir}/${exp_name}/cross_${split}/plan_${id}_temperature${temperature}.jsonl \
            --blocks ${blocks_dir}/cross_${split}  \
            --temperature ${temperature} \
            --port ${port} \
            --sem_limit ${sem_limit} --agent_family ${agent_family}  &
        done
        wait
    done
done




