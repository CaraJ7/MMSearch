import os
import json
from tqdm import tqdm

from utils.logging_utils import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)
import argparse

from utils.utils import *
from prompts.prompt import *
from prompts.prompt_w_imagesearch import *
from utils.prompt_utils import *
from models.load import load_model
from utils.image_utils import pil_image_to_bytes
from score.result_summary import get_result_summary
import datasets

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_type", default='Llava', type=str, help='the number of results from search engine')
    argparser.add_argument("--model_path", default='/data1/zrr/jdz/models/llava-next-interleave-qwen-7b', type=str, help='the number of results from search engine')
    argparser.add_argument("--world-size", type=int, default=1)
    argparser.add_argument("--rank", type=int, default=0)
    argparser.add_argument("--brief_result_num", default=8, type=int)
    argparser.add_argument("--fullpage_num", default=1, type=int)
    argparser.add_argument("--save_path", default='output/rerank/debug', type=str)
    argparser.add_argument("--generation_args_path", type=str, default='customs/generation_args.json', help='LMM generation parameters, should be a json')
    return argparser.parse_args()


args = parse_args()

sample_save_path = os.path.join(args.save_path, 'samples')
os.makedirs(sample_save_path, exist_ok=True)
# load model
model = load_model(args)

# load data
anno = datasets.load_dataset('CaraJ/MMSearch', name='rerank', split='rerank')
# calculate start and end for each rank
bin = len(anno) // args.world_size 
rank_start = bin * args.rank
rank_end = (args.rank+1)*bin if args.rank != args.world_size - 1 else len(anno)

brief_result_num = args.brief_result_num
fullpage_num = args.fullpage_num


result_list = []
for data_index, inst in tqdm(enumerate(anno)):
    # only run the instance for current rank
    if data_index < rank_start or data_index >= rank_end:
        continue

    # if this sample already exists, load the instance and continue
    if os.path.exists(os.path.join(sample_save_path, f"{inst['sample_id']}.json")):
        result_list.append(json.load(open(os.path.join(sample_save_path, f"{inst['sample_id']}.json"))))
        continue

    # if exists, load the instance and continue
    if inst['query_image'] is None:
        query_has_image = False
        prompt_template_dict = text_query_dict
    else:
        query_has_image = True
        prompt_template_dict = image_search_text_query_dict

    result_brief = [dict(
        **inst[f"website{i}_info"],
        screenshot_path=inst[f"website{i}_head_screenshot"]
    ) for i in range(brief_result_num)] # [{'title', 'text','screenshot_path', 'd'}]
    query = inst['query']

    website_information, input_image_list = get_website_information(result_brief)
    # the input image need to be converted to bytes for reading
    input_image_list = [pil_image_to_bytes(img) for img in input_image_list]
    
    # add query image
    prompt_template = prompt_template_dict['stage2'] 
    if not query_has_image:
        image_files = input_image_list
        text_query = prompt_template.format(
            brief_result_num=brief_result_num,
            rerank_num=fullpage_num,
            question=query,
            website_information=website_information,
            incontext_example=get_rerank_incontext_example(fullpage_num)
        )
    else:
        image_files = [
            pil_image_to_bytes(inst['query_image']),
            pil_image_to_bytes(inst['image_search_result']),
            *input_image_list
        ]
        text_query = prompt_template.format(
            brief_result_num=brief_result_num,
            rerank_num=fullpage_num,
            question=DEFAULT_IMAGE_TOKEN+query,
            image_search_result=DEFAULT_IMAGE_TOKEN,
            website_information=website_information,
            incontext_example=get_rerank_incontext_example(fullpage_num)
        )

    rerank = model.infer(
        image_files=image_files,
        text_query=text_query
    )

    selected_index, valid = postprocess_rerank(rerank, fullpage_num) 
    selected_index = selected_index[0] # only take the first one

    if not valid:
        score = 0
    elif selected_index in inst['valid']:
        score = 1
    elif selected_index in inst['not_sure']:
        score = 0.5
    else:
        score = 0

    save_inst = dict(
        sample_id=inst['sample_id'],
        query=inst['query'],
        model_output=rerank,
        model_output_valid=valid,
        parsed_answer_rank=selected_index,
        rer_score=score,
        valid=inst['valid'],
        not_sure=inst['not_sure'],
        invalid=inst['invalid'],
        area=inst['area'],
        subfield=inst['subfield'],
    )

    json.dump(save_inst, open(os.path.join(sample_save_path, f"{inst['sample_id']}.json"), 'w'), indent=4)
    result_list.append(save_inst)

result_summary = get_result_summary(anno, result_list, summary_key='rer_score')
logger.info(f"Total length: {result_summary['rer_score']['total_dict']['total_length']}")
logger.info(f"Average Rerank Score: {result_summary['rer_score']['total_dict']['average']}")
json.dump(
    result_summary, 
    open(os.path.join(args.save_path, f"result_summary_rerank.json"), 'w'), 
    indent=4
)
