'''
RUNNING THIS FILE WILL COMPLETE THE REQUERY TASK AT THE SAMETIME. 
'''

import os
import json
import datetime
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
from score.f1_score import get_f1_score
from utils.image_utils import pil_image_to_bytes
from retrieve_content.retriever import Content_Retriever
from constants import FULLPAGE_SPLIT_DICT
from score.result_summary import get_result_summary
import random


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_type", default='Llava', type=str, help='the number of results from search engine')
    argparser.add_argument("--model_path", default='lmms-lab/llava-onevision-qwen2-7b-ov', type=str, help='the number of results from search engine')
    argparser.add_argument("--world-size", type=int, default=1)
    argparser.add_argument("--rank", type=int, default=0)
    argparser.add_argument("--brief_result_num", default=8, type=int)
    argparser.add_argument("--fullpage_num", default=1, type=int)
    argparser.add_argument("--save_path", default='output/end2end/debug', type=str)
    argparser.add_argument("--save_middle_results", type=str, default='')
    argparser.add_argument("--generation_args_path", type=str, default='customs/generation_args.json', help='LMM generation parameters, should be a json')
    argparser.add_argument("--verbose", action='store_true', default=False,)
    return argparser.parse_args()
    
args = parse_args()

sample_save_path = os.path.join(args.save_path, 'samples')
os.makedirs(sample_save_path, exist_ok=True)

# load model
model = load_model(args)
# load content retriever
content_retriever = Content_Retriever()

# load data
anno = datasets.load_dataset('CaraJ/MMSearch', name='end2end', split='end2end')
# calculate start and end for each rank
bin = len(anno) // args.world_size 
rank_start = bin * args.rank
rank_end = (args.rank+1)*bin if args.rank != args.world_size - 1 else len(anno)

brief_result_num = args.brief_result_num
fullpage_num = args.fullpage_num

# setup dir
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Create a temporary directory with the timestamp
random.seed(args.rank)
if args.save_middle_results == '':
    temp_dir = f"temp_files/{timestamp}_{random.randint(1, 1000)}"
else:
    temp_dir = args.save_middle_results
os.makedirs(temp_dir, exist_ok=True)

'''
pipeline: 
1. stage1: GPT requery
2. stage2: GPT rerank
3. stage3: GPT summarization
'''

result_list = []
for data_index, inst in tqdm(enumerate(anno)):
    # only run the instance for current rank
    if data_index < rank_start or data_index >= rank_end:
        continue
    
    # if this sample already exists, load the instance and continue
    if os.path.exists(os.path.join(sample_save_path, f"{inst['sample_id']}.json")):
        result_list.append(json.load(open(os.path.join(sample_save_path, f"{inst['sample_id']}.json"))))
        continue

    # set up image dir
    screenshot_dir = f"{temp_dir}/{data_index}"
    os.makedirs(screenshot_dir, exist_ok=True)
    
    # prepare query information
    if inst['query_image'] is None:
        query_has_image = False
        prompt_template_dict = text_query_dict
    else: # query with image
        query_has_image = True
        prompt_template_dict = image_search_text_query_dict

    query = inst['query']

    logger.info('************************************* Stage1 *************************************')
    # input: query information
    # output: requery
    # then search the text
    stage1_screenshot_dir = os.path.join(screenshot_dir, 'stage1')
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
    if args.verbose:
        logger.info(f"Requery: {requery}")
        logger.info(f"Query ID: {inst['sample_id']}")

    result_brief = search_text_brief_result(
        query=requery, 
        max_result_num=brief_result_num, 
        screenshot_dir=stage1_screenshot_dir # relative path
    ) # [{'title', 'text','screenshot_path', 'url'}]

    if result_brief is None:
        logger.info("Duckduckgo returns None, skip this question")
        continue

    ### stage2: rerank
    logger.info('************************************* Stage2 *************************************')
    # input: query information
    # output: requery
    # then search the text
    website_information, input_image_list = get_website_information(result_brief)

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
    if args.verbose:
        logger.info(f"Rerank: {rerank}")

    ### stage3: gpt summarize
    logger.info('************************************* Stage3 *************************************')
    stage3_screenshot_dir = os.path.join(screenshot_dir, 'stage3')

    selected_index, _ = postprocess_rerank(rerank, fullpage_num)
    selected_website = [result_brief[i] for i in selected_index]
    result_full = search_url_full_result(
        urls=[web['url'] for web in selected_website], 
        screenshot_dir=stage3_screenshot_dir # relative path
    ) # [{'content', 'screenshot_fullpage_path'}]

    if args.verbose:
        logger.info(f'Selected index: {selected_index}')
        logger.info(selected_website)

    # add title and snippet
    for full_idx, brief_idx in enumerate(selected_index):
        result_full[full_idx]['title'] = result_brief[brief_idx]['title']
        result_full[full_idx]['snippet'] = result_brief[brief_idx]['snippet']

    # conduct content retrieval
    for idx, inst_full in enumerate(result_full):
        if inst_full['content'] is None: # in case cannot get web content
            inst_full['content'] = ''
        if inst_full['content'].strip() != '': # some web do not contain language content
            result_full[idx]['content'] = content_retriever.get_retrieved_content(requery, inst_full['content'])

    website_full_information, input_image_list = get_full_website_information(
        result_full=result_full,
        image_dir=stage3_screenshot_dir,
        fullpage_split_dict=FULLPAGE_SPLIT_DICT
    )
    
    # text_query and input_image_list
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
            rerank_num=args.fullpage_num,
            website_information=website_full_information,
            image_search_result=DEFAULT_IMAGE_TOKEN,
            question=DEFAULT_IMAGE_TOKEN + query
        )

    prediction = model.infer(
        image_files=image_files,
        text_query=text_query
    )
    if args.verbose:
        logger.info(f'Summarize: {prediction}')

    # calculate the score
    ## requery
    gt_requery = inst['gt_requery']
    req_score = get_requery_score(requery, gt_requery)
    ## end2end
    gt_answer = inst['gt_answer']
    f1_score = get_f1_score(prediction, gt_answer)
    for gt_alternative_answer in inst['alternative_gt_answers']:
        alternative_f1_score = get_f1_score(prediction, gt_alternative_answer)
        if alternative_f1_score > f1_score:
            f1_score = alternative_f1_score

    save_inst = dict(
        sample_id=inst['sample_id'],
        query=inst['query'],
        requery=requery,
        gt_requery=inst['gt_requery'],
        req_score=req_score['score'],
        req_score_dict=req_score,
        prediction=prediction,
        gt_answer=gt_answer,
        f1_score=f1_score,
        result_brief=result_brief,
        rerank=rerank,
        fullpage_url = [web['url'] for web in selected_website],
        result_full=result_full,
        area=inst['area'],
        subfield=inst['subfield'],
    )

    json.dump(save_inst, open(os.path.join(sample_save_path, f"{inst['sample_id']}.json"), 'w'), indent=4)
    result_list.append(save_inst)

result_summary = get_result_summary(anno, result_list, summary_key=['req_score', 'f1_score'])
logger.info(f"Total length: {result_summary['f1_score']['total_dict']['total_length']}")
logger.info(f"Average F1 Score: {result_summary['f1_score']['total_dict']['average']}")
json.dump(
    result_summary, 
    open(os.path.join(args.save_path, f"result_summary_end2end.json"), 'w'), 
    indent=4
)