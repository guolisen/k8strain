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
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os, wandb
from utils.prompter import Prompter

# ## 7. 模型合并

# In[ ]:


# 预训练模型
model_name = "D:/BaiduNetdiskDownload/2_LoRA课程资料/02_案例实战/Meta-Llama-3-8B-Instruct"


# In[ ]:


# 合并 base model 与 lora model
# https://huggingface.co/docs/trl/main/en/use_model#use-adapters-peft

base_model = AutoModelForCausalLM.from_pretrained(
    model_name, low_cpu_mem_usage=True,
    return_dict=True,torch_dtype=torch.float16,
    device_map= "auto")


# In[ ]:


new_model = PeftModel.from_pretrained(base_model, "D:/BaiduNetdiskDownload/lora-k8s-new12312")
#merged_model = base_model

# In[ ]:


# 模型合并

merged_model = new_model.merge_and_unload()
merged_model.eval()

# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

prompter = Prompter("llama3_data_format")
cutoff_len = 2048
columnMap = {"instruction": "instruction", "input": "input", "output": "output"}

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    ll = len(prompt)
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors="pt",
    )
    return result

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
    return tokenized_full_prompt
        
# In[ ]:

user_input = {"instruction": "Tell me something about the kubernetes.",
              "input": "",
              "output": ""}
#"Tell me something about the kubernetes."
#device = "cuda"
inputs = generate_and_tokenize_prompt(user_input)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
resp = merged_model.generate(**inputs, streamer=streamer, max_new_tokens=128, num_return_sequences=1)

print(resp)

# In[ ]:

