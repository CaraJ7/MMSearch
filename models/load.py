import json

def load_model(args):
    # load generation args
    if args.generation_args_path != None:
        generation_args = json.load(open(args.generation_args))

    if 'Llava_Onevision' in args.model_type:
        from models.llava_model import Llava_Onevision
        conv_mode = 'qwen_1_5'
        model = Llava_Onevision(model_path=args.model_path, conv_mode=conv_mode, generation_args=generation_args)
    else:
        raise NotImplementedError(f'{args.model_type} is not supported!')

    return model
