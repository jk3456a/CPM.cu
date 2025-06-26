'''
This script converts float model from MiniCPM format to Llama format.
'''
from transformers import AutoModelForCausalLM
import torch
import math
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from cpmcu.common.log_utils import logger
from tqdm import tqdm

def convert_llm(minicpm_path):

    model = AutoModelForCausalLM.from_pretrained(minicpm_path, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)

    scale_emb = model.config.scale_emb
    dim_model_base = model.config.dim_model_base
    scale_depth = model.config.scale_depth
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    logger.info(f"scale_emb = {scale_emb}")
    logger.info(f"dim_model_base = {dim_model_base}")
    logger.info(f"scale_depth = {scale_depth}")
    logger.info(f"num_layers = {num_layers}")
    logger.info(f"hidden_size = {hidden_size}")

    state_dict = model.state_dict()

    # model.embed_tokens.weight * scale_emb
    state_dict["model.embed_tokens.weight"] = state_dict["model.embed_tokens.weight"] * scale_emb

    # lm_head.weight / (hidden_size / dim_model_base)
    state_dict["lm_head.weight"] = state_dict["lm_head.weight"] / (hidden_size / dim_model_base)

    for i in tqdm(range(num_layers)):
        attn_out_name = f"model.layers.{i}.self_attn.o_proj.weight"
        state_dict[attn_out_name] = state_dict[attn_out_name] * (scale_depth / math.sqrt(num_layers))

        ffn_down_proj_name = f"model.layers.{i}.mlp.down_proj.weight"
        state_dict[ffn_down_proj_name] = state_dict[ffn_down_proj_name] * (scale_depth / math.sqrt(num_layers))

    torch.save(state_dict, os.path.join(minicpm_path, "pytorch_model.llama_format.bin"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=str, required=True, help="Path to MiniCPM model")
    args = parser.parse_args()

    convert_llm(args.src)
