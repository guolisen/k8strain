# -*- coding: utf-8 -*-
 
"""
Llama3微调结果推理
"""
 
# 服务器上运行时需要设置
import os

from transformers import AutoTokenizer as LLMTokenizer
from transformers import AutoModelForCausalLM as LLMModel
from transformers import (
    pipeline
)
from peft import PeftModel
import torch

CUDA_VISIBLE_DEVICES = "0"
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
device="cuda"

print("是否可用：", torch.cuda.is_available())        # 查看GPU是否可用
print("GPU数量：", torch.cuda.device_count())        # 查看GPU数量
print("torch方法查看CUDA版本：", torch.version.cuda)  # torch方法查看CUDA版本
print("GPU索引号：", torch.cuda.current_device())    # 查看GPU索引号
print("GPU名称：", torch.cuda.get_device_name(0))    # 根据索引号得到GPU名称

def generate_prompt(data_point, is_logger=False):

    text_input = data_point
    prompt_text_1 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Try to give solutions.<|eot_id|><|start_header_id|>user<|end_header_id|>
{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

    text_input_1 = prompt_text_1.format(text_input.strip())

    if is_logger:
        print(text_input_1)

    return text_input_1

#prompt_dict = generate_prompt("Can you think of Azure Resource Manager as the equivalent to what kubernetes is for Docker?")
prompt_dict = generate_prompt("If I start a Google Container Engine cluster like this:\ngcloud container clusters --zone=$ZONE create $CLUSTER_NAME\n\nI get three worker nodes.  How can I create a cluster with more?\n")   
def infer_only_lora():
    """
    只使用LoRA部分推理
    """
    new_model = "C:/code/llama/lora-k8s-third"
    tokenizer = LLMTokenizer.from_pretrained(new_model, trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    #model = LLMModel.from_pretrained(new_model, torch_dtype=torch.bfloat16)
    llama_pipeline = pipeline("text-generation", model=new_model, tokenizer=tokenizer)
    sentences = llama_pipeline(prompt_dict,
                               eos_token_id=tokenizer.eos_token_id, max_new_tokens=256)
    for seq in sentences:
        print(seq["generated_text"])
 
 
def infer_merge_llama_lora():
    """
    合并原始模型和LoRA后进行推理
    """
    # 基础模型路径
    base_model_path = "C:/code/llama/Meta-Llama-3-8B-Instruct"
    #base_model_path="D:/BaiduNetdiskDownload/2_LoRA课程资料/02_案例实战/Meta-Llama-3-8B-Instruct"

    # 加载基础模型
    base_model = LLMModel.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    # 加载tokenizer
    tokenizer = LLMTokenizer.from_pretrained(base_model_path, add_eos_token=True, trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # 微调模型路径
    new_model = "C:/code/llama/lora-k8s-forth"

    # 加载两者
    merge_model = PeftModel.from_pretrained(base_model, new_model)
    # 物理合并
    merge_model = merge_model.merge_and_unload()
    merge_model = merge_model.cuda()
 
    # Save model and tokenizer
    #output_model = "C:/code/llama/qu"
    #merge_model.save_pretrained(output_model)
    #tokenizer.save_pretrained(output_model)

    llama_pipeline = pipeline("text-generation", model=merge_model, tokenizer=tokenizer)
    sentences = llama_pipeline(prompt_dict,
                               eos_token_id=tokenizer.eos_token_id, max_new_tokens=256)
    for seq in sentences:
        print(seq["generated_text"])
 
 
if __name__ == '__main__':
    #infer_only_lora()
    infer_merge_llama_lora()