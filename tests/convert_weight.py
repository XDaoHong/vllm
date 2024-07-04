import os
import numpy as np
import torch
import argparse
import shutil

def load_model(model_dir):
    if not os.path.isdir(model_dir):
        raise ValueError(f"model dir:{model_dir} not exist.")

    stat_dict = dict()
    for file_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, file_name)
        if model_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            stat_dict.update(load_file(model_path, device="cpu"))
        elif model_path.endswith((".bin", ".pth", ".pt")):
            stat_dict.update(torch.load(model_path, map_location="cpu"))
    
    return stat_dict

def save_model(stat_dict, model_dir, safe_serialization=False):
    os.makedirs(model_dir, exist_ok=True)
    if safe_serialization:
        from safetensors.torch import save_file
        save_file(stat_dict, f"{model_dir}/model.safetensors", metadata={"format": "pt"})
    else:
        torch.save(stat_dict, f"{model_dir}/pytorch_model.bin")

def copy_files_with_prefix(src_dir, dst_dir, prefix):
    for file_name in os.listdir(src_dir):
        if file_name.startswith(prefix):
            src_file = os.path.join(src_dir, file_name)
            dst_file = os.path.join(dst_dir, file_name)
            shutil.copy2(src_file, dst_file)

def get_args():
    parser = argparse.ArgumentParser(description="convert weight parameters")
    parser.add_argument("--tp", 
                        type=int, 
                        required=True,
                        help="the number of tensor parallel")
    parser.add_argument("--layer_nums", 
                        type=int, 
                        required=True,
                        help="model layer nums")
    parser.add_argument("--origin_model_dir", 
                        type=str, 
                        default="./origin_model/", 
                        help="origin .safetensors or .bin model weight dir")
    parser.add_argument("--converted_model_dir", 
                        type=str, 
                        default="./convert_model/", 
                        help="converted .safetensors or .bin model weight dir")

    args, unknown_args = parser.parse_known_args()
    return args

def convert(args):
    stat_dict = load_model(args.origin_model_dir)
    ## Print model keys in the model
    # for key in stat_dict.keys():
    #     print(f"key:{key} value shape:{stat_dict[key].shape} dtype:{stat_dict[key].dtype}")

    new_dict = dict()
    keys_out_layers_list = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for key in keys_out_layers_list:
        new_dict[key] = stat_dict[key]

    for k in range(args.layer_nums):
        prefix = f"model.layers.{k}"
        m_names_list = ["self_attn", "mlp"]    # block names
        n_names_list = [["q_proj", "k_proj", "v_proj", "o_proj"],
                        ["gate_proj", "up_proj", "down_proj"]]  # matmul node names
        k_split_names = ["o_proj", "down_proj"] # this mm nodes will be splited at K dimension
        for i in range(len(m_names_list)):
            block_name = m_names_list[i]
            node_names = n_names_list[i]
            for name in node_names:
                key_name = f"{prefix}.{block_name}.{name}.weight"
                new_dict[key_name] = stat_dict[key_name]
                N = stat_dict[key_name].shape[0] # output channel
                K = stat_dict[key_name].shape[1] # input channel

                key_name = f"{prefix}.{block_name}.{name}.deq_scale"
                new_dict[key_name] = stat_dict[key_name]

                key_name = f"{prefix}.{block_name}.{name}.input_scale"
                new_dict[key_name.replace(f"input_scale", "scale_x")] = stat_dict[key_name].expand((K,)).to(torch.float32)

                key_name = f"{prefix}.{block_name}.{name}.input_offset"
                new_dict[key_name.replace(f"input_offset", "offset_x")] = stat_dict[key_name].expand((K,)).to(torch.int32)

                if name in k_split_names:
                    key_name = f"{prefix}.{block_name}.{name}.quant_bias"
                    bias = stat_dict[key_name]
                    zeros_bias = torch.zeros(N, dtype=torch.int32)

                    bias1 = bias // args.tp
                    bias2 = bias - (args.tp - 1) * bias1
                    for _ in range(args.tp - 1):
                        bias2 = torch.cat([bias2, bias1], axis=0)
                    new_dict[key_name.replace("quant_bias", "bias")] = bias2
                else:
                    key_name = f"{prefix}.{block_name}.{name}.quant_bias"
                    new_dict[key_name.replace("quant_bias", "bias")] = stat_dict[key_name]

        key_name = f"{prefix}.self_attn.rotary_emb.inv_freq"
        new_dict[key_name] = stat_dict[key_name]

        key_name = f"{prefix}.input_layernorm.weight"
        new_dict[key_name] = stat_dict[key_name]

        key_name = f"{prefix}.post_attention_layernorm.weight"
        new_dict[key_name] = stat_dict[key_name]


    save_model(new_dict, f"{args.converted_model_dir}")
    copy_files_with_prefix(args.origin_model_dir, args.converted_model_dir, "tokenizer")
    shutil.copy2(f"{args.origin_model_dir}/config.json", f"{args.converted_model_dir}/config.json")


if __name__ == "__main__":
    args = get_args()
    print(args)

    convert(args)