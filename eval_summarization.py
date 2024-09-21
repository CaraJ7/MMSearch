import os
import json
from tqdm import tqdm

from utils.logging_utils import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)
import argparse

from utils.utils import *
from prompts.prompt_w_imagesearch import *
from prompts.prompt import *
from utils.prompt_utils import *
from constants import *
from models.load import load_model
from score.f1_score import get_f1_score
from utils.image_utils import pil_image_to_bytes
from score.result_summary import get_result_summary
import datasets


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_type", default='Llava', type=str, help='the number of results from search engine')
    argparser.add_argument("--model_path", default='/data1/zrr/jdz/models/llava-next-interleave-qwen-7b', type=str, help='the number of results from search engine')
    argparser.add_argument("--world-size", type=int, default=1)
    argparser.add_argument("--rank", type=int, default=0)
    argparser.add_argument("--save_path", default='output/summarization/debug', type=str)
    argparser.add_argument("--generation_args_path", type=str, default='customs/generation_args.json', help='LMM generation parameters, should be a json')
    return argparser.parse_args()
    
args = parse_args()

sample_save_path = os.path.join(args.save_path, 'samples')
os.makedirs(sample_save_path, exist_ok=True)
# load model
model = load_model(args)

# load data
anno = datasets.load_dataset('CaraJ/MMSearch', name='summarization', split='summarization')
# calculate start and end for each rank
bin = len(anno) // args.world_size 
rank_start = bin * args.rank
rank_end = (args.rank+1)*bin if args.rank != args.world_size - 1 else len(anno)

fullpage_num = 1

result_list = []
for data_index, inst in tqdm(enumerate(anno)):
    # only run the instance for current rank
    if data_index < rank_start or data_index >= rank_end:
        continue
    
    if inst['query_image'] is None:
        continue

    # if this sample already exists, load the instance and continue
    if os.path.exists(os.path.join(sample_save_path, f"{inst['sample_id']}.json")):
        result_list.append(json.load(open(os.path.join(sample_save_path, f"{inst['sample_id']}.json"))))
        continue

    # set up prompt
    if inst['query_image'] is None:
        query_has_image = False
        prompt_template_dict = text_query_dict
    else:
        query_has_image = True
        prompt_template_dict = image_search_text_query_dict

    result_full = [dict(
        title=inst['website_title'],
        snippet=inst['website_snippet'],
        content=inst['website_retrieved_content'],
        slimmed_website_fullpage_screenshot=pil_image_to_bytes(inst['website_fullpage_screenshot']),
    )] # the screenshot from the dataset has already been slimmed

    website_full_information, input_image_list = get_full_website_information(
        result_full=result_full,
        fullpage_split_dict=FULLPAGE_SPLIT_DICT
    )
    query = inst['query']

    # add query image in the input image files
    prompt_template = prompt_template_dict['stage3']  
    if not query_has_image:
        image_files = input_image_list
        text_query = prompt_template.format(
            rerank_num=fullpage_num,
            website_information=website_full_information,
            question=query,
        )
    else:
        image_files = [
            *input_image_list,
            pil_image_to_bytes(inst['image_search_result']),
            pil_image_to_bytes(inst['query_image'])
        ]
        # assume only 1 image in the query
        text_query = prompt_template.format(
            rerank_num=fullpage_num,
            website_information=website_full_information,
            image_search_result=DEFAULT_IMAGE_TOKEN,
            question=DEFAULT_IMAGE_TOKEN + query
        )

    prediction = model.infer(
        image_files=image_files,
        text_query=text_query
    )

    gt_answer = inst['gt_answer']
    f1_score = get_f1_score(prediction, gt_answer)

    save_inst = dict(
        sample_id=inst['sample_id'],
        query=inst['query'],
        prediction=prediction,
        gt_answer=gt_answer,
        f1_score=f1_score,
        area=inst['area'],
        subfield=inst['subfield']
    )

    json.dump(save_inst, open(os.path.join(sample_save_path, f"{inst['sample_id']}.json"), 'w'), indent=4)
    result_list.append(save_inst)

result_summary = get_result_summary(anno, result_list, summary_key='f1_score')
logger.info(f"Total length: {result_summary['f1_score']['total_dict']['total_length']}")
logger.info(f"Average f1_score: {result_summary['f1_score']['total_dict']['average']}")
json.dump(
    result_summary, 
    open(os.path.join(args.save_path, f"result_summary.json"), 'w'), 
    indent=4
)
