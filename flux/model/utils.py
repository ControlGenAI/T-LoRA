import torch
from transformers import PretrainedConfig, T5TokenizerFast
from diffusers.utils.torch_utils import is_compiled_module


def get_layer_by_name(model, mname):
    """
    Extract layer using layer name
    """
    module = model
    mname_list = mname.split(".")
    for mname in mname_list:
        module = module._modules[mname]
    return module


def load_text_encoders(args, class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    )
    return text_encoder_one, text_encoder_two


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model