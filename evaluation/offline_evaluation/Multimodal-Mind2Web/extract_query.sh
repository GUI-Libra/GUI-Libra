

directory='path/to/your/directory'
for split in 'website' 'task' 'domain'
do 
    temperature=0.0
    for index in 0 #2 3 4 5
    do
        python extract_grounding_query.py --input_file data/${directory}/cross_${split}/plan_${index}_temperature${temperature}.jsonl \
        --output_file data/${directory}/cross_${split}/query_${index}_temperature${temperature}.jsonl \
        --blocks path/to/blocks_images/cross_${split}  
    done
done


