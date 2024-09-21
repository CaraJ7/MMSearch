export PW_TEST_SCREENSHOT_NO_FONTS_READY=1
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python eval_summarization.py \
--model_type Llava_Onevision \
--model_path lmms-lab/llava-onevision-qwen2-7b-ov \
--generation_args_path customs/generation_args.json \
--save_path 'output/summarization/llava' 