export PW_TEST_SCREENSHOT_NO_FONTS_READY=1
export OPENAI_API_KEY=sk-8Fc4olxBbLImGvjg15F81b93CbA14e459bD5Ac7232718aD2
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python eval_requery.py \
--rank 0 \
--world-size 1 \
--model_type gpt-4o-mini-2024-07-18 \
--model_path /data1/zrr/jdz/models/llava-onevision-qwen2-7b-ov \
--save_path 'output/requery/gpt4omini' 
