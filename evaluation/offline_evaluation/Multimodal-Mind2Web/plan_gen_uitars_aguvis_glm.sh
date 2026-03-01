dir='evaluation'
add_template=1
reasoning=1
blocks_dir='path/to/blocks_images'

sem_limit=100
ports=(8002 8003 8004)

for split in  'domain'   'task'  'website'
do 
    for temperature in 0.0
    do
        for id in 0 
        do
            exp_name='GLM-4.1V-9B-Thinking'
            python aguvis_uitars_plan.py \
                --agent_family glm \
                --model zai-org/GLM-4.1V-9B-Thinking \
                --port 8002 \
                --sem_limit ${sem_limit} \
                --input_file data/samples/cross_${split}_blocks_natural.jsonl \
                --output_file data/${dir}/${exp_name}/cross_${split}/plan_${id}_temperature${temperature}.jsonl \
                --blocks ${blocks_dir}/cross_${split}   &


            exp_name='Aguvis-7B-720P'
            python aguvis_uitars_plan.py \
                --agent_family aguvis \
                --model xlangai/Aguvis-7B-720P \
                --port 8003 \
                --sem_limit ${sem_limit} \
                --input_file data/samples/cross_${split}_blocks_natural.jsonl \
                --output_file data/${dir}/${exp_name}/cross_${split}/plan_${id}_temperature${temperature}.jsonl \
                --blocks ${blocks_dir}/cross_${split}  


            exp_name='UI-TARS-1.5-7B'
            python aguvis_uitars_plan.py \
                --agent_family uitars \
                --uitars_template computer \
                --model ByteDance-Seed/UI-TARS-1.5-7B \
                --port 8004 \
                --sem_limit ${sem_limit} \
                --input_file data/samples/cross_${split}_blocks_natural.jsonl \
                --output_file data/${dir}/${exp_name}/cross_${split}/plan_${id}_temperature${temperature}.jsonl \
                --blocks ${blocks_dir}/cross_${split}  &


        done
    done
    wait

     temperature=1.0
     for id in 0 1 2 3 
     do 
            exp_name='GLM-4.1V-9B-Thinking'
            python aguvis_uitars_plan.py \
                --agent_family glm \
                --model zai-org/GLM-4.1V-9B-Thinking \
                --port 8002 \
                --sem_limit ${sem_limit} \
                --input_file data/samples/cross_${split}_blocks_natural.jsonl \
                --output_file data/${dir}/${exp_name}/cross_${split}/plan_${id}_temperature${temperature}.jsonl \
                --blocks ${blocks_dir}/cross_${split} &

            
            exp_name='Aguvis-7B-720P'
            python aguvis_uitars_plan.py \
                --agent_family aguvis \
                --model xlangai/Aguvis-7B-720P \
                --port 8003 \
                --sem_limit ${sem_limit} \
                --input_file data/samples/cross_${split}_blocks_natural.jsonl \
                --output_file data/${dir}/${exp_name}/cross_${split}/plan_${id}_temperature${temperature}.jsonl \
                --blocks ${blocks_dir}/cross_${split}  &

            exp_name='UI-TARS-1.5-7B'
            python aguvis_uitars_plan.py \
                --agent_family uitars \
                --uitars_template computer \
                --model ByteDance-Seed/UI-TARS-1.5-7B \
                --port 8004 \
                --sem_limit ${sem_limit} \
                --input_file data/samples/cross_${split}_blocks_natural.jsonl \
                --output_file data/${dir}/${exp_name}/cross_${split}/plan_${id}_temperature${temperature}.jsonl \
                --blocks ${blocks_dir}/cross_${split}  &  
     
     wait   
     done
    

done

        