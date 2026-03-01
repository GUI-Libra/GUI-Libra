

model_path="osunlp/UGround-V1-7B"
sufix='uground'
image_dir='path/to/AndroidControl_images'

directory='path/to/plan_directory'
for key in 'high' 'low'
do 
    for temperature in 0.0
    do
        for index in 0
        do 
            question_file=data/${directory}/query_${key}_${index}_temperature${temperature}.jsonl
            answer_file=data/${directory}/ans_${key}_${index}_${sufix}_temperature${temperature}.jsonl

            python grounding/uground_qwen2vl_serve.py --model-path ${model_path} --question-file ${question_file} \
                                --answers-file ${answer_file}  --image-folder ${image_dir}  \
                                --temperature 0.0 --image-key image --port 8000
        done
    done
done
