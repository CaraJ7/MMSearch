#!/bin/bash

WORLDSIZE="8"

rank=0
list=(0 1 2 3 4 5 6 7)

for i in "${list[@]}"
do

    PW_TEST_SCREENSHOT_NO_FONTS_READY=1 \
    CUDA_VISIBLE_DEVICES=$i \
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    RANK=$i \
    WORLD_SIZE=8 \
    python eval_rerank.py \
    --rank $i \
    --world-size 8 \
    --model_type Llava_Onevision \
    --model_path lmms-lab/llava-onevision-qwen2-7b-ov \
    --generation_args_path customs/generation_args.json \
    --save_path 'output/rerank/llava'  &
   
   rank=$((rank + 1))
done
