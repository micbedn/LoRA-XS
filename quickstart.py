from peft import LoraConfig, get_peft_model
from utils.initialization_utils import find_and_initialize  # used to transform LoRA to LoRA-XS 

config = LoraConfig(
    r=lora_rank,
    target_modules=lora_target_modules,
    task_type="CAUSAL_LM", # assuming a decoder-only model in this example
        )
model = get_peft_model(model, config)

with open("config/reconstruct_config.yaml", 'r') as stream:
    reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)
    
adapter_name = "default"  # assuming a single LoRA adapter per module should be transformed to LoRA-XS
peft_config_dict = {adapter_name: lora_config}

# specifying LoRA rank for the SVD initialization
reconstr_config['svd']['rank'] = lora_rank
    
find_and_initialize(
    model, peft_config_dict, adapter_name=adapter_name, reconstr_type='svd',
    writer=None, reconstruct_config=reconstr_config
    )

# perform training...

# LoRA-XS can be merged into the base model using `merge_and_unload` functionality of PEFT
model = model.merge_and_unload() 


