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
from utils.image_utils import *
from prompts.prompt import *
from prompts.prompt_w_imagesearch import *
from utils.prompt_utils import *
from models.load import load_model
from score.req_score import get_requery_score
from score.f1_score import get_f1_score
from retrieve_content.retriever import Content_Retriever
from constants import FULLPAGE_SPLIT_DICT
from score.result_summary import get_result_summary
import random


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_type", default='Llava', type=str, help='the number of results from search engine')
    argparser.add_argument("--model_path", default='lmms-lab/llava-onevision-qwen2-7b-ov', type=str, help='the number of results from search engine')
    argparser.add_argument("--brief_result_num", default=8, type=int)
    argparser.add_argument("--fullpage_num", default=1, type=int)
    argparser.add_argument("--query_json", default='demo/query_cli.json', type=str)
    argparser.add_argument("--save_path", default='output/demo_cli/debug', type=str)
    argparser.add_argument("--save_middle_results", type=str, default='')
    argparser.add_argument("--generation_args_path", type=str, default='customs/generation_args.json', help='LMM generation parameters, should be a json')
    return argparser.parse_args()
    
args = parse_args()

sample_save_path = os.path.join(args.save_path, 'samples')
os.makedirs(sample_save_path, exist_ok=True)

# load query
query_list = json.load(open(args.query_json))
# load model
model = load_model(args)
# load content retriever
content_retriever = Content_Retriever()

brief_result_num = args.brief_result_num
fullpage_num = args.fullpage_num
assert fullpage_num == 1 # we currently only support 1 fullpage input

# setup dir
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Create a temporary directory with the timestamp
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
for data_index, inst in tqdm(enumerate(query_list)):
    inst['sample_id'] = data_index
    
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
    # conduct image search process, ensure the opened chrome is English
    if query_has_image:
        if inst.get('image_search_result', None) is None or not os.path.exists(inst['image_search_result']):
            image_url = inst['query_image_url']
            save_image_search_result_path = os.path.join(screenshot_dir, f"image_search_result.jpg")
            image_search_result = search_by_image(image_url, save_image_search_result_path) # this return the list of each items, we do not use it
            crop_image_search_results(
                image_path = save_image_search_result_path, 
                save_path = save_image_search_result_path
            )
            inst['image_search_result'] = save_image_search_result_path

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
            inst['query_image'],
            inst['image_search_result'],
        ]
        text_query = prompt_template.format(
            question=DEFAULT_IMAGE_TOKEN + query,
            image_search_result=DEFAULT_IMAGE_TOKEN
        )

    requery = model.infer(
        image_files=image_files,
        text_query=text_query
    )
    logger.info(f"Requery: {requery}")

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
            inst['query_image'],
            inst['image_search_result'],
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
            inst['image_search_result'],
            inst['query_image']
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
    logger.info(f'Answer: {prediction}')

    

    save_inst = dict(
        sample_id=inst['sample_id'],
        query=inst['query'],
        requery=requery,
        prediction=prediction,
        result_brief=result_brief,
        rerank=rerank,
        fullpage_url=[web['url'] for web in selected_website],
        result_full=result_full,
    )

    json.dump(save_inst, open(os.path.join(sample_save_path, f"{inst['sample_id']}.json"), 'w'), indent=4)
    result_list.append(save_inst)

json.dump(result_list, open(os.path.join(sample_save_path, f"result_summary.json"), 'w'), indent=4)