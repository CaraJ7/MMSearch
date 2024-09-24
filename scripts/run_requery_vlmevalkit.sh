export PW_TEST_SCREENSHOT_NO_FONTS_READY=1
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python eval_requery.py \
--rank 0 \
--world-size 1 \
--model_type vlmevalkit_llava_onevision_qwen2_7b_ov \
--save_path 'output/requery/vlmevalkit_llava' 
