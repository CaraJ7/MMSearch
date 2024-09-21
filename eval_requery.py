'''
RUNNING THIS FILE WILL COMPLETE THE REQUERY TASK AT THE SAMETIME. 
'''

import os
import json
import datasets
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
from score.req_score import get_requery_score
from utils.image_utils import pil_image_to_bytes
from score.result_summary import get_result_summary


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_type", default='Llava', type=str, help='the number of results from search engine')
    argparser.add_argument("--model_path", default='lmms-lab/llava-onevision-qwen2-7b-ov', type=str, help='the number of results from search engine')
    argparser.add_argument("--world-size", type=int, default=1)
    argparser.add_argument("--rank", type=int, default=0)
    argparser.add_argument("--save_path", default='output/requery/debug', type=str)
    argparser.add_argument("--generation_args_path", type=str, default='customs/generation_args.json', help='LMM generation parameters, should be a json')
    return argparser.parse_args()
    
args = parse_args()

sample_save_path = os.path.join(args.save_path, 'samples')
os.makedirs(sample_save_path, exist_ok=True)

# load model
model = load_model(args)

# load data
anno = datasets.load_dataset('CaraJ/MMSearch', name='end2end', split='end2end')
# calculate start and end for each rank
bin = len(anno) // args.world_size 
rank_start = bin * args.rank
rank_end = (args.rank+1)*bin if args.rank != args.world_size - 1 else len(anno)


result_list = []
for data_index, inst in tqdm(enumerate(anno)):
    # only run the instance for current rank
    if data_index < rank_start or data_index >= rank_end:
        continue

    # if this sample already exists, load the instance and continue
    if os.path.exists(os.path.join(sample_save_path, f"{inst['sample_id']}.json")):
        result_list.append(json.load(open(os.path.join(sample_save_path, f"{inst['sample_id']}.json"))))
        continue
    
    # prepare query information
    if inst['query_image'] is None:
        query_has_image = False
        prompt_template_dict = text_query_dict
    else: # query with image
        query_has_image = True
        prompt_template_dict = image_search_text_query_dict

    query = inst['query']

    prompt_template = prompt_template_dict['stage1']  
    if not query_has_image:
        image_files = []
        text_query = prompt_template.format(question=query)
    else:
        image_files = [
            pil_image_to_bytes(inst['query_image']),
            pil_image_to_bytes(inst['image_search_result'])
        ]
        text_query = prompt_template.format(
            question=DEFAULT_IMAGE_TOKEN + query,
            image_search_result=DEFAULT_IMAGE_TOKEN
        )

    requery = model.infer(
        image_files=image_files,
        text_query=text_query
    )
    
    # calculate the score
    ## requery
    gt_requery = inst['gt_requery']
    req_score = get_requery_score(requery, gt_requery)

    save_inst = dict(
        sample_id=inst['sample_id'],
        query=inst['query'],
        requery=requery,
        gt_requery=inst['gt_requery'],
        req_score=req_score['score'],
        req_score_dict=req_score,
        area=inst['area'],
        subfield=inst['subfield'],
    )

    json.dump(save_inst, open(os.path.join(sample_save_path, f"{inst['sample_id']}.json"), 'w'), indent=4)
    result_list.append(save_inst)

result_summary = get_result_summary(anno, result_list, summary_key='req_score')
logger.info(f"Total length: {result_summary['req_score']['total_dict']['total_length']}")
logger.info(f"Average Rerank Score: {result_summary['req_score']['total_dict']['average']}")
json.dump(
    result_summary, 
    open(os.path.join(args.save_path, f"result_summary_requery.json"), 'w'), 
    indent=4
)