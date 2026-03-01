split=website task domain


block_path='path_to/UGround-Offline-Evaluation/Multimodal-Mind2Web_blocks_images/release_images'

plan_file=path/to/your/plan/file

ans_file=path/to/your/ans/file

python eval.py --sample_file data/samples/cross_${split}_blocks.jsonl \
          --plan_file ${plan_file} \
           --ans_file  ${ans_file}  --blocks ${block_path}

