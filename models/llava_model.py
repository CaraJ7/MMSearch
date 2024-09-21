import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import copy

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List

from PIL import Image
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class Llava_Onevision:
    def __init__(self, model_path, conv_mode, generation_args):
        disable_torch_init()
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        # from https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_OneVision_Tutorials.ipynb
        llava_model_args = {
            "multimodal": True,
        }
        overwrite_config = {}
        overwrite_config["image_aspect_ratio"] = "pad"
        llava_model_args["overwrite_config"] = overwrite_config
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base=None, model_name=model_name, **llava_model_args)

        self.device = 'cuda'
        self.conv_mode = conv_mode
        self.generation_args = generation_args
        self.model.eval()
        self.model.tie_weights()
        for _, p in self.model.named_parameters():
            p.requires_grad = False

    # for onevision interleaved
    def infer(self, image_files, text_query):
        qs = text_query
        # Prepare interleaved text-image input
        conv = copy.deepcopy(conv_templates[self.conv_mode])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

        image_tensors = []
        if len(image_files) != 0:
            images = [Image.open(f).convert("RGB") for f in image_files]
            image_tensors = process_images(images, self.image_processor, self.model.config)
            image_tensors = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensors]
            image_sizes = [image.size for image in images]


        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensors if len(image_tensors) != 0 else None,
                image_sizes=image_sizes if len(image_tensors) != 0 else None,
                do_sample=True if self.generation_args.get('temperature', 0) > 0 else False,
                temperature=self.generation_args.get('temperature', 0),
                top_p=self.generation_args.get('top_p', None),
                num_beams=self.generation_args.get('num_beams', 1),
                max_new_tokens=self.generation_args.get('max_new_tokens', 256),
                use_cache=True)

        
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        return outputs
