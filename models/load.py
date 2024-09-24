import json

def load_model(args):
    # load generation args
    if args.generation_args_path != None:
        generation_args = json.load(open(args.generation_args_path))
    # load model
    ## use custom defined model
    if 'Llava_Onevision' in args.model_type: 
        from models.llava_model import Llava_Onevision
        conv_mode = 'qwen_1_5'
        model = Llava_Onevision(model_path=args.model_path, conv_mode=conv_mode, generation_args=generation_args)
    ## use models implemented from VLMEvalKit. You need to first install VLMEvalKit from https://github.com/open-compass/VLMEvalKit
    ## the available name list of the model is https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py
    ## You can replace your model path to the file above
    ## Note that, some model in VLMEvalKit do not support text-only inference, so it may not support end2end task (some queries in round1 do not have image input).
    elif 'vlmevalkit_' in args.model_type: 
        from models.vlmevalkit_model_api import VLMEvalModel
        model = VLMEvalModel(model_type=args.model_type)
    else:
        raise NotImplementedError(f'{args.model_type} is not supported!')

    return model
