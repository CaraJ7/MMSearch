import os
import json
import argparse
from score.result_summary import get_result_summary
import datasets

from utils.logging_utils import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)

task_key_dict = dict(
    end2end='f1_score',
    requery='req_score',
    rerank='rer_score',
    summarization='f1_score',
)

task_ratio_dict = dict(
    end2end=0.75,
    requery=0.05,
    rerank=0.1,
    summarization=0.1,
)

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--save_path", default='result_summary_final.json', type=str)
    argparser.add_argument("--end2end_sample_dir", type=str)
    argparser.add_argument("--requery_sample_dir", type=str)
    argparser.add_argument("--rerank_sample_dir", type=str),
    argparser.add_argument("--summarization_sample_dir", type=str)
    return argparser.parse_args()
    
args = parse_args()

anno = datasets.load_dataset('CaraJ/MMSearch', name='end2end', split='end2end')

all_task_result_summary = dict()
for task, key in task_key_dict.items():
    task_result_list = [
        json.load(open(os.path.join(getattr(args, f"{task}_sample_dir"), f))) 
        for f in os.listdir(getattr(args, f"{task}_sample_dir"))
    ]
    all_task_result_summary[task] = get_result_summary(anno, task_result_list, key)[key]

# total dict
final_result_summary = dict()
final_result_summary['total_dict'] = dict()
final_result_summary['total_dict']['average'] = sum(
    [ratio*all_task_result_summary[task]['total_dict']['average'] for task, ratio in task_ratio_dict.items()]
)
# area dict
final_result_summary['area_dict'] = dict()
for area in all_task_result_summary['end2end']['area_dict']:
    final_result_summary['area_dict'][area] = sum(
    [ratio*all_task_result_summary[task]['area_dict'][area]['average'] for task, ratio in task_ratio_dict.items()]
)
# subfield dict
final_result_summary['subfield_dict'] = dict()
for subfield in all_task_result_summary['end2end']['subfield_dict']:
    final_result_summary['subfield_dict'][subfield] = sum(
    [ratio*all_task_result_summary[task]['subfield_dict'][subfield]['average'] for task, ratio in task_ratio_dict.items()]
)

logger.info(f"Average final score: {final_result_summary['total_dict']['average']}")
json.dump(final_result_summary, open(args.save_path, 'w'))