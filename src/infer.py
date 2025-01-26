import json
import os
import torch
import transformers
import openai
import time
import argparse
import yaml

import re
import time
import tiktoken
import concurrent.futures
import shortuuid
import tqdm
import random


def read_json(json_path, type="read"):
    with open(json_path, 'r') as f:
        if type == "load":
            content = json.load(f)
        elif type == "read":
            content = f.readlines()
        elif type == "read_str":
            data_str = f.read()
            content = json.loads(data_str)
        else:
            print("type error")
            sys.exit()
    return content


# API setting constants
API_MAX_RETRY = 10000 #16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"
def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None):
    import openai
    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                )
            output = completion.choices[0].message.content
            assert len(output) > 0
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except KeyError:
            print(type(e), e)
            # break
        except Exception as e:
            print(type(e), e)
            # break
    
    return output

# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs

def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(
        endpoint_list
    )[0]
    return api_dict

def get_answer(
    question: str, model: str, endpoint_info: dict, max_tokens: int, temperature: float, api_dict: dict
):
    api_type = endpoint_info["api_type"]

    conv = []

    if "system_prompt" in endpoint_info.keys():
        conv.append({"role": "system", "content": endpoint_info["system_prompt"]})


    matches = re.findall(r'([AQ]\d[:：])(.*?)((?=[\n][AQ]\d[:：])|$)', question, re.DOTALL)
    if matches:
        multi_turns_num = len(matches)
    else:
        multi_turns_num = 0

    if multi_turns_num == 0:
        conv.append({"role": "user", "content": question})
        if api_type == "openai":
            output = chat_completion_openai(model=endpoint_info["model_name"], 
                                            messages=conv, 
                                            temperature=temperature, 
                                            max_tokens=max_tokens, 
                                            api_dict=api_dict)
            
        else:
            output = "Error: api_type != openai"


        return output
    else:
        output = ""
        for index, i_q in enumerate(matches):
            conv.append({"role": "user", "content": i_q[1]})
            if api_type == "openai":
                this_output = chat_completion_openai(model=endpoint_info["model_name"], 
                                                messages=conv, 
                                                temperature=temperature, 
                                                max_tokens=max_tokens, 
                                                api_dict=api_dict)
                
            else:
                this_output = "Error: api_type != openai"
            
            conv.append({"role": "assistant", "content": this_output})
            output += f"A{index+1}: {this_output}\n"
        return output



def run_qwen_o1_vllm(input_file, endpoint_info, settings, output_prefix=None):
    """
    Run our o1 from qwen2.5 based on vllm
    :param 
    """

    dataset_list = [json.loads(data) for data in read_json(input_file)]
    
    for model in settings["model_list"]:
        assert model in endpoint_list
        if output_prefix is None:
            output_path = os.path.join("results/long-cot", model, input_file.split('/')[-1][:-6],"infer_results.jsonl")
        else:
            output_path = os.path.join(output_prefix, model, input_file.split('/')[-1][:-6],"infer_results.jsonl")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        endpoint_info = endpoint_list[model]
        
        for data_dict in dataset_list:
            instruction = data_dict["query"]
            # messages = [{"role": "user", "content": instruction}]
            response = get_answer(instruction, model, endpoint_info, max_tokens=settings['max_tokens'], temperature=settings['temperature'], api_dict=get_endpoint(endpoint_info["endpoints"]))
            
            output_dict = {"id": data_dict["id"], "query": instruction, "answer": response, "meta": data_dict["meta"]}  # , "checklists": data_dict["checklists"], "formatted_checklists": data_dict["formatted_checklists"]
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def process_task(data_dict, model, endpoint_info, settings, output_prefix, input_file):

    if output_prefix is None:
        output_path = os.path.join("results/long-cot", model, input_file.split('/')[-1][:-6],"infer_results.jsonl")
    else:
        output_path = os.path.join(output_prefix, model, input_file.split('/')[-1][:-6],"infer_results.jsonl")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    instruction = data_dict["query"]
    response = get_answer(instruction, model, endpoint_info, max_tokens=settings['max_tokens'], temperature=settings['temperature'], api_dict=get_endpoint(endpoint_info["endpoints"]))
    
    output_dict = {"id": data_dict["id"], "query": instruction, "answer": response, "meta": data_dict["meta"]}

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")

def process_all_tasks(dataset_list, model, endpoint_info, settings, output_prefix, input_file, max_threads=8):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(process_task, i_data, model, endpoint_info, settings, output_prefix, input_file) for i_data in dataset_list]
        for future in futures:
            future.result()  # wait for all tasks to complete

def run_qwen_o1_vllm_parallel(input_file, endpoint_info, settings, output_prefix, max_threads=8):
    """
    Run our o1 from qwen2.5 based on vllm
    :param 
    """
    dataset_list = [json.loads(data) for data in read_json(input_file)]

    for model in settings["model_list"]:
        assert model in endpoint_list
        endpoint_info = endpoint_list[model]
        process_all_tasks(dataset_list, model, endpoint_info, settings, output_prefix, input_file, max_threads=max_threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file", type=str, default="./dataset/test.jsonl"
    )
    parser.add_argument(
        "--setting-file", type=str, default="src/configs/vllm/gen_answer_config.yaml"
    )
    parser.add_argument(
        "--endpoint-file", type=str, default="src/configs/vllm/api_config.yaml"
    )
    parser.add_argument(
        "--output_prefix", type=str, default=None
    )
    parser.add_argument(
        "--parallel", type=int, default=0
    )
    args = parser.parse_args()

    settings = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)



    if args.parallel == 0:
        run_qwen_o1_vllm(args.input_file, endpoint_list, settings, args.output_prefix)  # , model_type="chat"
    else:
        run_qwen_o1_vllm_parallel(args.input_file, endpoint_list, settings, args.output_prefix, args.parallel)

