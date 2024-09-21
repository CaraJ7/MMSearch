export PW_TEST_SCREENSHOT_NO_FONTS_READY=1
export OPENAI_API_KEY=sk-8Fc4olxBbLImGvjg15F81b93CbA14e459bD5Ac7232718aD2
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python eval_end2end.py \
--rank 0 \
--world-size 1 \
--model_type gpt-4o-mini-2024-07-18 \
--model_path /data1/zrr/jdz/models/llava-onevision-qwen2-7b-ov \
--brief_result_num 8 \
--fullpage_num 1 \
--save_path 'output/end2end/gpt4omini' \
--save_middle_results middle_results/debug
