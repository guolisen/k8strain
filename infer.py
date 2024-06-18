# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/25 21:56
# @author  : Mo
# @function: 推理


import traceback
import random
import time
import sys
import os

CUDA_VISIBLE_DEVICES = "1"
USE_TORCH = "1"
CPU_NUMS = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TORCH"] = USE_TORCH
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model
from peft import LoraConfig, PeftModel
from transformers import GenerationConfig
from tensorboardX import SummaryWriter
from datasets import load_dataset
from pydantic import BaseModel
from rouge import Rouge  # pip install rouge
from tqdm import tqdm
import torch

from transformers import AutoTokenizer as LLMTokenizer
from transformers import AutoModelForCausalLM as LLMModel
from transformers import LlamaConfig as LLMConfig

PATH_MODEL_PRETRAIN="C:/code/llama/Meta-Llama-3-8B-Instruct"
#MODEL_SAVE_DIR
TARGET_MODULES = ["q_proj",
                  #"k_proj",
                  "v_proj",
                  # "o_proj",
                  # "down_proj",
                  # "gate_proj",
                  # "up_proj",
                  ]
USE_CACHE=True
USE_CUDA=True
MAX_LENGTH_Q = 2048
LORA_DROPOUT = 0.05
LORA_ALPHA = 16
LORA_R = 8

def print_named_parameters(model, use_print_data=True):
    """   打印模型训练参数/数据类型信息   """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if use_print_data:
            print((name, param.data.dtype, param.requires_grad, param.data))
        else:
            print((name, param.data.dtype, param.requires_grad))
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
def generate_prompt(data_point, is_logger=False):
    """   指令微调:
    普通句子续写: bos + text + eos
    带 prompt:
    """
    text_input = data_point.get("input", "")
    text_out = data_point.get("output", "")
#     prompt_text_1 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#
# You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
#
# {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
#     你是一个电商广告创意大师, 请用简体中文写广告创意.
    prompt_text_1 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant, answer questions and give solutions.<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
    text_prompt = tokenizer.encode(prompt_text_1, add_special_tokens=False)
    text_input_1 = prompt_text_1.format(text_input.strip())
    x = tokenizer.encode(text_input_1, add_special_tokens=False)
    if len(x) > MAX_LENGTH_Q:
        x = x[:MAX_LENGTH_Q-len(text_prompt)]
    out = {"input_ids": x, "labels": []}
    if is_logger:
        print(text_input_1)
        print(text_prompt)
        print(out)
    return out

print("是否可用：", torch.cuda.is_available())        # 查看GPU是否可用
print("GPU数量：", torch.cuda.device_count())        # 查看GPU数量
print("torch方法查看CUDA版本：", torch.version.cuda)  # torch方法查看CUDA版本
print("GPU索引号：", torch.cuda.current_device())    # 查看GPU索引号
print("GPU名称：", torch.cuda.get_device_name(0))    # 根据索引号得到GPU名称

tokenizer = LLMTokenizer.from_pretrained(PATH_MODEL_PRETRAIN,
                                         add_eos_token=True,
                                         trust_remote_code=True)


tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"  

"""
<|begin_of_text|> [128000]
<|start_header_id|> [128006]
system [9125]
<|end_header_id|> [128007]
<|eot_id|> [128009]
user [882]
<|end_header_id|> [128007]
assistant [78191]
\n\n [271]
\n [198]
"""
#STOP_WORDS_IDS = [[ID_BOS], [ID_EOS], [ID_END]]


# llm_config = LLMConfig.from_pretrained(PATH_MODEL_PRETRAIN)
# model = LLMModel(llm_config)
# model.init_weights()
# model = model.half()
base_model = LLMModel.from_pretrained(PATH_MODEL_PRETRAIN, torch_dtype=torch.bfloat16)
# model = LLMModel.from_pretrained(PATH_MODEL_PRETRAIN, torch_dtype=torch.float32)

# model = prepare_model_for_half_training(model,
#         use_gradient_checkpointing=True,
#         output_embedding_layer_name="lm_head",
#         layer_norm_names=["post_attention_layernorm",
#                           "input_layernorm",
#                           "norm"
#                           ],
#         )
# model.gradient_checkpointing_enable()
# model.enable_input_require_grads()
# model.gradient_checkpointing_disable()
# model.is_parallelizable = IS_PARALLELIZABLE
# model.model_parallel = MODEL_PARALLEL
base_model.config.use_cache = USE_CACHE

new_model = PeftModel.from_pretrained(base_model, "C:/code/llama/lora-k8s-third")
#new_model = PeftModel.from_pretrained(base_model, "C:/code/llama/lora-k8s-second")
model = new_model.merge_and_unload()
#model = base_model

#print_named_parameters(model)
if USE_CUDA:
    model = model.cuda()

model.eval()
# for param in filter(lambda p: p.requires_grad, model.parameters()):
#     param.data = param.data.to(torch.float16)

# for name, param in model.named_parameters():
#     if "LoR" in name:   # 某些peft版本默认dtype=fp16, 这里全部转为 fp32
#         param.data = param.data.to(torch.float32)

#print_named_parameters(model)


def predict(data_dict):
    """  推理  """
    prompt_dict = generate_prompt(data_dict)
    # inputs = tokenizer([text_1], return_tensors="pt", padding=True)
    input_ids = prompt_dict.get("input_ids")
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    if USE_CUDA:
        input_ids = input_ids.cuda()
    generation_config = GenerationConfig(
        # temperature=0.8,
        # top_p=0.8,
        temperature=0.9,
        top_p=0.75,
        do_sample=True,
        max_new_tokens=512,
        # penalty_alpha=1.5,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            #stop_words_ids=STOP_WORDS_IDS,
            #stop_words_ids=[[tokenizer.eos_token_id]],
            return_dict_in_generate=True,
            # return_dict_in_generate=True,
            output_scores=True,
            # max_new_tokens=512,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print(data_dict)
    print(input_ids)
    print(output)
    # output = output.split("答：")[-1]
    return output



if __name__ == '__main__':
    data_dict = {"input": "Tell me something about the kubernetes.",
                 }
    res = predict(data_dict)
    print(res)
    while True:
        time_start = time.time()
        history = []
        print("请输入:")
        ques = input()
        print("请稍等...")
        try:
            print("#" * 128)
            ques_dict = {"input": ques}
            res = predict(ques_dict)
            print(res)
        except Exception as e:
            print(str(e))
        print(time.time() - time_start)
