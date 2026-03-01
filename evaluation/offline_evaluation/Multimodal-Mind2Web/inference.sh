

model_path="osunlp/UGround-V1-7B"
key='cross_domain'


for key in 'cross_website' 'cross_task' 'cross_domain'
do 
    question_file='offline_evaluation/Multimodal-Mind2Web/data/gpt-4o_results/'${key}'_query.jsonl'
    answer_file='offline_evaluation/Multimodal-Mind2Web/data/gpt-4o_results/'${key}'_answers.jsonl' # path to save the answers
    image_dir='path/to/blocks_images'

    python grounding/uground_qwen2vl_serve.py --model-path ${model_path} --question-file ${question_file} \
                        --answers-file ${answer_file}  --image-folder ${image_dir}/${key}  \
                        --temperature 0.0 --image-key image
done

model_path="osunlp/UGround-V1-7B"
for key in 'cross_website'  'cross_task' 'cross_domain'
do 
    for index in 0 1 2 3
    do 
        question_file=offline_evaluation/Multimodal-Mind2Web/data/o4-mini_repeat/${key}/query_${index}.jsonl
        answer_file=offline_evaluation/Multimodal-Mind2Web/data/o4-mini_repeat/${key}/answers_${index}_aguvis.jsonl # path to save the answers
        image_dir='path/to/blocks_images'

        python grounding/aguvis_qwen2vl_serve.py --model-path ${model_path} --question-file ${question_file} \
                            --answers-file ${answer_file}  --image-folder ${image_dir}/${key}  \
                            --temperature 0.0 --image-key image
    done
done

