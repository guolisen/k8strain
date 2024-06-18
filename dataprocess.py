# -*- coding: utf-8 -*-

 
import json
import csv
from utils.prompter import Prompter
from datasets import load_dataset

def generate_and_tokenize_prompt(data_point):
    prompter = Prompter("llama3_data_format")
    columnMap = {"instruction": "instruction", "input": "input", "output": "output"}
    full_prompt = prompter.generate_prompt(
        data_point[columnMap["instruction"]],
        data_point[columnMap["input"]],
        data_point[columnMap["output"]],
    )
    #tokenized_full_prompt = tokenize(full_prompt)
    return full_prompt
        
def process(): 
    #data = load_dataset("dh02391735/stackoverflow-kubernetes-questions")
    #data["train"].to_json('./datasets/datasets.json')

    input_filename = "./datasets/datasets.json"
    output_filename = "./datasets/datasets_llama3.json"
    
    result_json_list = []
    with open(output_filename, "w") as write_file:
        with open(input_filename, 'r') as read_file:
            # 将文件对象转换为reader对象
            #reader=csv.reader(read_file)
            
            # 循环遍历reader对象
            '''
            for item in reader:
                each_result_json = {"text": generate_and_tokenize_prompt(item)}
                result_json_list.append(each_result_json)
            '''
            data_json = json.load(read_file)
            for each_json in data_json:
                each_result_json = {"text": generate_and_tokenize_prompt(each_json)}
                result_json_list.append(each_result_json)


            write_file.write(json.dumps(result_json_list, ensure_ascii=True, indent=4))

    print ("Done")
 
if __name__ == '__main__':
    process()