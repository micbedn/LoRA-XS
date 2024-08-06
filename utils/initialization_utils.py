import math
import types

import peft
import torch
from peft.import_utils import is_bnb_available
from peft.utils import _get_submodules
from torch.nn import init
from tqdm import tqdm

from .latent_utils import get_delta_weight, forward_latent
from .svd_utils import get_linear_rec_svd


def get_replacement_module(weight, module_name, type, writer, reconstruct_config):
    print("get_replacement_module:")
    print(f"module_name(key): {module_name}")
    print(f"weight.shape: {weight.shape}")
    print(f"type: {type}")
    print(f"reconstruct_config: {reconstruct_config}")
    cfg = reconstruct_config[type]
    print(f"cfg: {cfg}")

    rank = cfg['rank']
    print(f"cfg[rank]: {cfg['rank']}")
    print(f"rank: {rank}")

    #change rank in layer 0
    if module_name == 'base_model.model.roberta.encoder.layer.1.attention.self.query':
        cfg['rank'] = 8
        print(f"rank in {module_name} changed to:{cfg['rank']}")

    if type == 'svd':
        reconstructed_matrix, enc, dec = get_linear_rec_svd(weight.cpu().detach().numpy(), cfg['rank'],
        #reconstructed_matrix, enc, dec = get_linear_rec_svd(weight.cpu().detach().numpy(), rank,
                                                            cfg['n_iter'],
                                                            cfg['random_state'])
        final_enc = torch.tensor(enc, dtype=weight.dtype, device=weight.device)
        final_dec = torch.tensor(dec, dtype=weight.dtype, device=weight.device)
    else:
        raise NotImplementedError(f"{type} is currently not supported.")
    
    cfg['rank'] = rank
    print("final_enc.shape: ")
    print(final_enc.shape)
    print("final_dec.shape: ")
    print(final_dec.shape)
    return final_enc, final_dec


def init_module_weights(target_module: torch.nn.Linear, sigma: float):
    # Initialize weights with Gaussian distribution
    torch.nn.init.normal_(target_module.weight, mean=0, std=sigma)
    if hasattr(target_module, "bias"):
        # Set bias to zeros
        if target_module.bias is not None:
            torch.nn.init.zeros_(target_module.bias)


def replace_module_weights(target_module, new_weight):
    device = target_module.weight.device
    target_module.weight = torch.nn.Parameter(new_weight)

    # dispatch to correct device
    for name, module in target_module.named_modules():
        if "lora_" in name:
            module.to(device)


def update_decoder_weights(target_module, new_weight):
    device = target_module.weight.device
    with torch.no_grad():
        target_module.weight.copy_(new_weight)

    # dispatch to correct device
    for name, module in target_module.named_modules():
        if "lora_" in name:
            module.to(device)


def kaiming_uniform_init_lower_half(matrix: torch.tensor):
    rows, _ = matrix.size()
    init.kaiming_uniform_(matrix[math.ceil(rows / 2):, :], a=math.sqrt(5))
    return matrix

def kaiming_uniform_init(matrix: torch.tensor):
    init.kaiming_uniform_(matrix, a=math.sqrt(5))
    return matrix
  
def find_and_initialize(model, peft_config, adapter_name, reconstr_type, reconstruct_config, writer):
    """
    :param adapter_name: options: 'default'
    :param reconstr_type: options: 'svd'
    """
    #peft_config = LoraConfig(
    #    task_type="SEQ_CLS",
    #    inference_mode=False,
    #    r=model_args.lora_rank,
    #    lora_alpha=model_args.lora_alpha,
    #    lora_dropout=0.0,
    #    target_modules=["query", "value", "attention.output.dense", "output.dense"],
    #)
    #model = get_peft_model(model, peft_config)

    half_init_dec = reconstruct_config['half_init_dec'] #False

    replacement_module_random_init = reconstruct_config['replacement_module_random_init'] #False

    reconstruction_mode = reconstruct_config['reconstr_mode'] #'separated'

    lora_config = peft_config[adapter_name]
    #LoraConfig(
    #    task_type="SEQ_CLS",
    #    inference_mode=False,
    #    r=model_args.lora_rank,
    #    lora_alpha=model_args.lora_alpha,
    #    lora_dropout=0.0,
    #    target_modules=["query", "value", "attention.output.dense", "output.dense"],
    #)

    r_squared = reconstruct_config['r_squared']  # whether using r*r matrix between lora_A and lora_B or not
    #True

    loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
    if loaded_in_8bit and not is_bnb_available():
        raise ImportError(
            "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
            "You can install it with `pip install bitsandbytes`."
        )
    is_target_modules_in_base_model = False

    # model.named_modules() returns an iterator over all modules in the network, yielding both the name of the module as well as the module itself.
    key_list = [key for key, _ in model.named_modules()]
    print()
    print("key_list aka model.named_modules()")
    for i in key_list:
        print(i)
    print()
    print()
    print()
    print(f"lora_config.target_modules: {lora_config.target_modules}")
    print(f"lora_config: {lora_config}")
    print(f"lora_config.rank_pattern: {lora_config.rank_pattern}")
    #exit()
    assert (not isinstance(lora_config.target_modules, str))
    print("Iterating through model's specified modules to initialize A/B matrices.")
    for key in tqdm(key_list):
        target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
        if target_module_found:
            if not is_target_modules_in_base_model:
                is_target_modules_in_base_model = True
            _, target, target_name = _get_submodules(model, key)
            print(f"key: {key}")
            print(f"target_name: {target_name}")
            print(f"target: {target}")
            print("_:", _)

            if reconstruction_mode == 'separated':
                replacement_encoder_weight, replacement_decoder_weight = get_replacement_module(weight=target.weight.T,
                                                                                                module_name=key,
                                                                                                type=reconstr_type,
                                                                                                writer=writer,
                                                                                                reconstruct_config=reconstruct_config)

                if not isinstance(target, peft.tuners.lora.Linear):
                    raise NotImplementedError('Only initialization for peft.tuners.lora.Linear type is implemented.')
                    # TODO implement for Linear8bitLt
                else:
                    if half_init_dec:
                        kaiming_uniform_init_lower_half(replacement_decoder_weight)
                    if replacement_module_random_init:
                        kaiming_uniform_init(replacement_encoder_weight)
                        kaiming_uniform_init(replacement_decoder_weight)
                    replace_module_weights(target.lora_B.default, replacement_decoder_weight.T)
                    if r_squared:
                        target.forward = types.MethodType(forward_latent, target)
                        target.get_delta_weight = types.MethodType(get_delta_weight, target)
                        replace_module_weights(target.lora_A.default, replacement_encoder_weight.T)

                        #if key.endswith("layer.1.attention.self.query"):
                        if key == "base_model.model.roberta.encoder.layer.1.attention.self.query":
                            lora_config.r = 8
                        else:
                            lora_config.r = 4 #reset to default

                        target.default_lora_latent_mapping = torch.nn.Linear(lora_config.r, lora_config.r, bias=False)

                        init_module_weights(target.default_lora_latent_mapping, sigma=0.00001)
                        target.default_lora_latent_mapping.to(target.lora_A.default.weight.device)

                        target.lora_A.default.weight.requires_grad = False  # only the r*r matrix will be tuned
                        target.lora_B.default.weight.requires_grad = False  # only the r*r matrix will be tuned

                    else:
                        init_module_weights(target.lora_A.default, sigma=0.00001)

            else:
                raise NotImplementedError("The only supported mode is: separated.")
            print("lora config.r:", lora_config.r)
            print("AFTER MODIFICATION")
            print(f"key: {key}")
            print(f"target_name: {target_name}")
            print(f"target: {target}")
            print("_:", _)
            _, target, target_name = _get_submodules(model, key)
            print("AFTER _get_submodules")
            print(f"key: {key}")
            print(f"target_name: {target_name}")
            print(f"target: {target}")
            print("_:", _)
            print()
            print()
            print()

    if not is_target_modules_in_base_model:
        raise ValueError(
            f"Target modules {lora_config.target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )
