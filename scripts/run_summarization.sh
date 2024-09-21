export PW_TEST_SCREENSHOT_NO_FONTS_READY=1
export OPENAI_API_KEY=sk-8Fc4olxBbLImGvjg15F81b93CbA14e459bD5Ac7232718aD2
export CUDA_VISIBLE_DEVICES=7
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python eval_summarization.py \
--model_type gpt-4o \
--model_path /data1/zrr/jdz/models/llava-next-interleave-qwen-7b \
--save_path 'output/summarization/gpt4omini' 