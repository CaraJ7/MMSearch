export PW_TEST_SCREENSHOT_NO_FONTS_READY=1
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python demo/mmsearch_engine.py \
--model_type Llava_Onevision \
--model_path lmms-lab/llava-onevision-qwen2-7b-ov \
--generation_args_path customs/generation_args.json \
--save_path 'demo_output/llava' \
--save_middle_results demo_middle_results/llava 