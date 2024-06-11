#!/usr/bin/env python
# coding: utf-8

import subprocess
import sys
import os
import fire
from typing import List
from utils.prompter import Prompter

# bitsandbytes：专为量化设计的库，重点在于减少大语言模型（尤其是在GPU上）的内存占用。
# peft：用于将LoRA适配器集成到大语言模型（LLMs）中。
# trl：该库包含一个SFT（监督微调）类，用于辅助微调模型。
# accelerate和xformers：这些库用于提高模型的推理速度，从而优化其性能。
# wandb：该工具作为一个监控平台，用于跟踪和观察训练过程。
# datasets：与Hugging Face一起使用，该库便于加载数据集。

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    TextStreamer,
    Trainer
)
from trl import SFTTrainer
import os, wandb
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    #prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

def train(
    # model/data params
    base_model: str = "D:/BaiduNetdiskDownload/2_LoRA课程资料/02_案例实战/Meta-Llama-3-8B-Instruct",  # the only required argument
    data_path: str = "dh02391735/stackoverflow-kubernetes-questions",
    output_dir: str = "./lora-k8s/",
    # training hyperparams
    batch_size: int = 2,
    micro_batch_size: int = 1,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    cutoff_len: int = 2048,
    val_set_size: int = 22800,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "finetune k8s kb test",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "llama3_data_format",  # The prompt template to use, will default to alpaca.
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        f"prompt template: {prompt_template_name}\n")

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)
    columnMap = {"instruction": "instruction", "input": "input", "output": "output"}
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if torch.cuda.is_available():
        device = "cuda"
        print(device)
    
    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0
    #if use_wandb:
    #    wandb.login(key="01449db30f1efc5720bc0afb5cf15b876a670407")
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_API_KEY"] = '01449db30f1efc5720bc0afb5cf15b876a670407'
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        ll = len(prompt)
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point[columnMap["instruction"]],
            data_point[columnMap["input"]],
            data_point[columnMap["output"]],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        '''
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point[columnMap["instruction"]], data_point[columnMap["input"]]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        '''

        return tokenized_full_prompt

    # Load module
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True, 
        bnb_4bit_quant_type = "nf4",  
        bnb_4bit_compute_dtype = torch.float16, 
        bnb_4bit_use_double_quant = False, 
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config = bnb_config,
        device_map = device_map
    )
    model = prepare_model_for_kbit_training(model) 
    #model = prepare_model_for_int8_training(model)

    # tokenizer 加载
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.padding_side = "left"  # Allow batched inference

    peft_config = LoraConfig(
        r=lora_r, # 8
        lora_alpha=lora_alpha, # 16
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout, # 0.05
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=2023
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs = num_epochs,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = gradient_accumulation_steps, 
        optim = "paged_adamw_8bit", # optim="adamw_torch",
        logging_steps = 30,
        learning_rate = learning_rate,
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        weight_decay = 0.001, # 权重衰减系数，用于L2正则化，帮助防止过拟合。
        fp16 = False,
        bf16 = False,
        max_grad_norm = 0.3, # 最大梯度范数，用于梯度裁剪，防止梯度爆炸。
        max_steps = -1, # 最大训练步数为-1，表示没有限制。
        eval_steps=32 if val_set_size > 0 else None,
        save_steps=32, # 每100步保存一次模型
        save_total_limit=5,
        load_best_model_at_end=True if val_set_size > 0 else False,
        warmup_ratio = 0.1, # 预热阶段的比例。在训练开始时，学习率会逐渐升高，预热比例为0.3表示前30%的训练步骤用于预热。
        group_by_length = group_by_length, # 按序列长度分组，以提高训练效率。
        lr_scheduler_type = "linear", # 表示使用线性学习率调度。
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
    )

    trainer = SFTTrainer(
        model = model,
        train_dataset = train_data,
        peft_config = peft_config,
        tokenizer = tokenizer,
        dataset_text_field="prompt",
        args = training_arguments,
        packing=False
    )

    model.config.use_cache = False
    '''
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    '''

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    #model.save_pretrained(output_dir)
    trainer.model.save_pretrained(output_dir)
    wandb.finish()

    model.config.use_cache = True
    print("\n Done!!!!")


if __name__ == "__main__":
    fire.Fire(train)