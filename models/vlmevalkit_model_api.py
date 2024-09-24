import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import copy
import hashlib
from PIL import Image
import logging
from constants import *

logger = logging.getLogger(__name__)

from vlmeval.config import supported_VLM

import re

logger = logging.getLogger(__name__)

os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)

def count_image_tags_and_text(text):
    pattern = r'(<image>+)'
    parts = re.split(pattern, text)
    
    result = []
    for part in parts:
        if part.startswith('<image>'):
            # num of <image>  
            count = len(part) // 7  #  len(<image>) = 7
            result.append(count)
        else:
            # text part
            result.append(part)
    
    # if start with <image>
    if text.startswith('<image>'):
        result.insert(0, 0)
    
    return result

class VLMEvalModel:
    def __init__(self, model_type):
        model_name = model_type.split('vlmevalkit_')[-1]
        self.model = supported_VLM[model_name](max_new_tokens=512)

    # for onevision interleaved
    def infer(self, image_files, text_query):
        '''
        image_files: a list of image file path or bytes (could be directly loaded with Image.open()). The order of the image files is the same order to LMM.
        text_query: the instruction to the LMM. We use '<image>' to denote the place for an image.
        '''
        # save image to disk to accommodate VLMEVALKIT_API
        save_name = hashlib.md5(text_query.encode("utf8")).hexdigest()
        save_dir = os.path.join(SAVE_IMAGE_DIR, save_name)
        os.makedirs(save_dir, exist_ok=True)
        for idx, f in enumerate(image_files):
            if isinstance(f, str):
                continue
            else:
                image_path = os.path.join(save_dir, f"{idx}.png")
                Image.open(f).save(image_path)
                image_files[idx] = image_path
        # split text_query to vlmevalkit struct [dict(type='image', value=IMAGE_PTH), dict(type='text', value=prompt)]
        query_input = []
        # insert the image in the text
        result = count_image_tags_and_text(text_query)
        image_idx = 0
        for trunk in result:
            # image input
            if isinstance(trunk, int):
                input_image_list = image_files[image_idx:image_idx+trunk]
                image_idx = image_idx + trunk
                for image_path in input_image_list:
                    query_input.append(
                        {
                            "type": "image",
                            "value": image_path
                        }
                    )
            # text input
            elif trunk.strip() != '':
                query_input.append(
                    {"type": "text", "value": trunk}
                )
        response = self.model.generate(message=query_input, dataset='MathVerse_MINI_Vision_Only') # the dataset here is only for not throwing an error

        return response
