# -*- coding: utf-8 -*-
 
"""
Llama3 Lora PEFT
"""
 
# 服务器上运行时需要设置
import time
import os
from typing import List

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
 
import torch
 
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
 
 
def peft_fine_tune():
    # 基础模型路径
    base_model_path = "/root/autodl-tmp/Meta-Llama-3-8B-Instruct"
    # 数据集路径
    #kb_dataset = "./datasets/datasets_llama3.json"
    kb_dataset = "sidddddddddddd/kubernetes-llama3"
    output_dir = "./lora-k8s-third/"
    # 采用json格式的数据集加载方式
    #dataset = load_dataset("json", data_files=kb_dataset, split="train")
    dataset = load_dataset(kb_dataset, split="train")
    #print(dataset)
    '''
    val_set_size = 0
    if val_set_size > 0:
        train_val = dataset["text"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=2023
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    '''
    # 用于线性层计算的数据类型
    compute_dtype = getattr(torch, "float16")
    # 量化参数
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 启用 4 位加载
        bnb_4bit_quant_type="nf4",  # 指定用于量化的数据类型。支持两种量化数据类型： fp4 （四位浮点）和 nf4 （常规四位浮点）
        bnb_4bit_compute_dtype=compute_dtype,  # 用于线性层计算的数据类型
        bnb_4bit_use_double_quant=False  # 是否使用嵌套量化来提高内存效率
    )
 
    wandb_project = "finetune k8s kb third"
    wandb_watch = ""  # options: false | gradients | all
    wandb_log_model = ""  # options: false | true
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_API_KEY"] = '01449db30f1efc5720bc0afb5cf15b876a670407'
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if torch.cuda.is_available():
        device = "cuda"
        print(device)

    use_bf16 = False
    if torch.cuda.is_bf16_supported():
        use_bf16 = True

    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        low_cpu_mem_usage=True,
        quantization_config = quant_config,
        torch_dtype = torch.bfloat16
    )
    base_model = prepare_model_for_kbit_training(base_model) 
    # use_cache是对解码速度的优化，在解码器解码时，存储每一步输出的hidden-state用于下一步的输入
    # 因为后面会开启gradient checkpoint，中间激活值不会存储，因此use_cahe=False
    base_model.config.use_cache = False
    # 设置张量并行
    base_model.config.pretraining_tp = 1
 
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    # 指定填充标记(pad_token)使用结束标记(eos_token)。pad_token是tokenizer中用于补足输入序列长度的填充标记,默认是 [PAD]。
    # eos_token是tokenizer中用于表示序列结束的标记,默认是 [SEP]
    tokenizer.pad_token = tokenizer.eos_token
    # padding_side 设置为“right”以修复 fp16 的问题
    # train的时候需要padding在右边，并在句末加入eos，否则模型永远学不会什么时候停下
    # test的时候需要padding在左边，否则模型生成的结果可能全为eos
    tokenizer.padding_side = "right"
    tokenizer.add_eos_token = True

    # LoRA微调参数
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ]
    peft_params = LoraConfig(
        lora_alpha=16,  # LoRA超参数，用于缩放低秩适应的权重
        lora_dropout=0.05,  # LoRA层的丢弃率
        r=8,  # LoRA中的秩
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM"  # Llama属于因果语言模型
    )
    peft_model = get_peft_model(base_model, peft_params)
    peft_model.print_trainable_parameters()
    peft_model = peft_model.cuda()

    # 训练器参数
    training_params = TrainingArguments(
        output_dir=output_dir,  # 结果路径
        num_train_epochs=3,  # 总的训练轮数
        per_device_train_batch_size=2,  # 这是每个GPU的训练批次大小
        gradient_accumulation_steps=1,  # 累积多个步骤的梯度，以有效地增加批次大小
        gradient_checkpointing=True,  # 模型支持梯度检查点
        gradient_checkpointing_kwargs={"use_reentrant": False},  # 解决use_reentrant警告
        optim="paged_adamw_8bit",  # 优化器
        save_steps=30,  # 保存检查点之间的步数
        save_total_limit=5,
        logging_steps=50,  # 训练日志输出之间的步数
        learning_rate=2e-4,  # 初始学习率
        weight_decay=0.001,  # 权重衰减率
        bf16=use_bf16 ,  # 不启用BF16
        fp16=False,  # 不启用混合精度训练
        max_grad_norm=0.3,  # 裁剪梯度
        #max_steps=1000,  # 最大训练迭代次数
        warmup_ratio=0.03,  # 训练开始时的预热样本比例
        group_by_length=True,  # 将训练数据集中大致相同长度的样本分组到同一batch中，提升prefill效率
        lr_scheduler_type="linear",  # 学习率调度器将使用常数衰减策略
        report_to=["tensorboard"]  # 将指标记录到Tensorboard
    )
 
    # 训练器
    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=dataset,
        peft_config=peft_params,
        dataset_text_field="text",  # 数据集中用于训练的文本字段
        max_seq_length=1024,  # 序列长度
        tokenizer=tokenizer,
        args=training_params,
        packing=False,  # 不将多个权重参数打包成更少的数据单元进行存储和传输
    )
 
    print("开始训练")
    start_time = time.time()
    trainer.train()
    trainer.save_model()
    end_time = time.time()
    print("训练结束")
    print("耗时：", end_time - start_time)
 
 
if __name__ == '__main__':
    peft_fine_tune()