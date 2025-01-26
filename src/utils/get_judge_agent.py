from ast import arg
import pickle
import sys
import os
import json
import yaml
import numpy as np 
import re
import math
import tiktoken

from .utils import extract_info2, logger, save_csv,response_long_check

#for new api format
from .api_general_v2 import generalv2_sub_chat

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENC=tiktoken.get_encoding('cl100k_base')
#读取环境变量
DANLUNKEGUAN_MAX_TOKEN =  int(os.getenv('DANLUNKEGUAN_MAX_TOKEN', 120000))  #7123  # 8024 0 1024
MULTITURN_MAX_TOKEN = int(os.getenv('MULTITURN_MAX_TOKEN', 120000))  #6611  # 8024 0 1024


############################################### vis and save
def agent_danlunkeguan_visfunc(autoprompt_list, instruct_list, res_prompt, prompt_ab,output_path, prefix="单轮客观"):
    ##
    os.makedirs(os.path.join('/'.join(output_path.split('/')[:-1]), 'analysis'), exist_ok=True)
    if ".jsonl" in output_path.split('/')[-1]:
        csv_path = os.path.join('/'.join(output_path.split('/')[:-1]), 'analysis', output_path.split('/')[-1].replace(".jsonl",f"_{prefix}analysis.csv"))
    else:
        csv_path = os.path.join('/'.join(output_path.split('/')[:-1]), 'analysis', output_path.split('/')[-1].replace(".json",f"_{prefix}analysis.csv"))
        
    ##解析和保存结果， jsonl and csv
    task_definition_statistic = {}
    task_definition_statistic['人工总分'] = 0
    task_definition_statistic['自动化总分'] = 0
    task_definition_statistic['分数不一致率'] = 0
    task_definition_statistic['分数不一致率±1'] = 0
    task_definition_statistic["无法判断"] = 0

    # 
    index_res_prompt = 0
    count_work = 0
    answer_str = None
    for i, (i_data, i_veri) in enumerate(zip(autoprompt_list, instruct_list)):
        auto_prompt = prompt_ab[i]

        if 'model_answer' in i_veri:
            answer_str = 'model_answer'
        elif 'answer' in i_veri:
            answer_str = 'answer'
        else:
            raise NotImplementedError

        if res_prompt[i] >= 0:
            predict_score = res_prompt[i]
            tmp_score = i_veri.get('gt', np.nan)
            if type(tmp_score) is str:
                tmp_score = float(tmp_score)
                # print(f"convert tmp_score to float from str: {tmp_score}")
            task_definition_statistic['人工总分'] += tmp_score #i_veri['gt']
            task_definition_statistic['自动化总分'] += predict_score
            if predict_score != tmp_score: #i_veri.get('gt', np.nan): #i_veri['gt']:
                task_definition_statistic['分数不一致率'] += 1
            
            if abs(predict_score - tmp_score) > 1:
                task_definition_statistic['分数不一致率±1'] += 1
            
            count_work += 1
        else:
            predict_score = -1
            task_definition_statistic["无法判断"] += 1
 
        instruct_list[i]['auto_prompt'] = auto_prompt
        instruct_list[i]['auto_score'] = predict_score

    ## save csv
    save_csv(csv_path, instruct_list, task_definition_statistic, keys=['id', 'query', 'reference', 'score_standard',answer_str, 'auto_prompt', 'gt', 'auto_score'])

    ## write json
    count_work = max(1, count_work)
    instruct_list.insert(0, {"人工总分": task_definition_statistic["人工总分"], "人工总分归一化": task_definition_statistic["人工总分"]/count_work, "自动化总分": task_definition_statistic["自动化总分"], "自动化总分归一化": task_definition_statistic["自动化总分"]/count_work, "分数不一致率": task_definition_statistic["分数不一致率"] / count_work, "分数不一致率±1": task_definition_statistic["分数不一致率±1"] / count_work, "计数题目数": count_work, "无法判断": task_definition_statistic["无法判断"]})

    fout = open(csv_path.replace(".csv", ".jsonl"), 'w')
    for i_data in instruct_list:
        json.dump(i_data, ensure_ascii=False, fp=fout)
        fout.write("\n")
    fout.close()

    print(f"########## {prefix}题 - 模型：{output_path.split('/')[-1]}")
    print("###############人工总分归一化： {}/{} = {}".format(task_definition_statistic["人工总分"], count_work, task_definition_statistic["人工总分"]/count_work))
    print("###############自动化总分归一化： {}/{} = {}".format(task_definition_statistic["自动化总分"], count_work, task_definition_statistic["自动化总分"]/count_work))
    print("###############分数不一致率： {}/{} = {}".format(task_definition_statistic["分数不一致率"], count_work, task_definition_statistic["分数不一致率"]/count_work))
    print("###############分数不一致率±1： {}/{} = {}".format(task_definition_statistic["分数不一致率±1"], count_work, task_definition_statistic["分数不一致率±1"]/count_work))
    print("###############无法判断 {}".format(task_definition_statistic["无法判断"]))




def agent_multiturn_tool_visfunc(autoprompt_list, instruct_list, res_prompt, prompt_ab,output_path, print_prefix="多轮"):
    ##
    os.makedirs(os.path.join('/'.join(output_path.split('/')[:-1]), 'analysis'), exist_ok=True)
    if ".jsonl" in output_path.split('/')[-1]:
        csv_path = os.path.join('/'.join(output_path.split('/')[:-1]), 'analysis', output_path.split('/')[-1].replace(".jsonl",f"_{print_prefix}analysis.csv"))
    else:
        csv_path = os.path.join('/'.join(output_path.split('/')[:-1]), 'analysis', output_path.split('/')[-1].replace(".json",f"_{print_prefix}analysis.csv"))

    ##解析和保存结果， jsonl and csv
    task_definition_statistic = {}
    task_definition_statistic['人工总分'] = 0
    task_definition_statistic['自动化总分'] = 0
    task_definition_statistic['分数不一致率'] = 0
    task_definition_statistic['分数不一致率±1'] = 0
    task_definition_statistic["无法判断"] = 0

    #
    index_res_prompt = 0
    count_work = 0
    instruct_list_csv = []
    for i, (i_data, i_veri) in enumerate(zip(autoprompt_list, instruct_list)):
        if res_prompt[i] >= 0:
            predict_score = res_prompt[i]
            try:
                temp_score = float(i_veri.get('gt', np.nan))
            except ValueError:
                i_veri['gt'] = 0
                temp_score = 0
            task_definition_statistic['人工总分'] += temp_score 
            task_definition_statistic['自动化总分'] += predict_score
            if predict_score != i_veri.get('gt', np.nan): 
                task_definition_statistic['分数不一致率'] += 1

            if abs(predict_score - temp_score) > 1:
                task_definition_statistic['分数不一致率±1'] += 1
            
            count_work += 1
        else:
            predict_score = -1
            task_definition_statistic["无法判断"] += 1
        
        auto_prompt = [] 
        for num, i_prompt in enumerate(i_data['auto_prompt']):
            auto_prompt.append(prompt_ab[index_res_prompt])

            instruct_list_csv.append(instruct_list[i].copy())
            instruct_list_csv[index_res_prompt]['auto_prompt'] = prompt_ab[index_res_prompt]
            if num > 0:
                instruct_list_csv[index_res_prompt]['id'] = ""
                instruct_list_csv[index_res_prompt]['query'] = ""
                instruct_list_csv[index_res_prompt]['gt'] = ""
            else:
                instruct_list_csv[index_res_prompt]['auto_score'] = predict_score
            index_res_prompt += 1
        
        instruct_list[i]['auto_prompt'] = auto_prompt
        instruct_list[i]['auto_score'] = predict_score

    ## save csv
    save_csv(csv_path, instruct_list_csv, task_definition_statistic, keys=['id', 'query', 'reference', 'score_standard',"model_answer", 'auto_prompt', 'gt', 'auto_score'])

    ## write json
    instruct_list.insert(0, {"人工总分": task_definition_statistic["人工总分"], "人工总分归一化": task_definition_statistic["人工总分"]/count_work, "自动化总分": task_definition_statistic["自动化总分"], "自动化总分归一化": task_definition_statistic["自动化总分"]/count_work, "分数不一致率": task_definition_statistic["分数不一致率"] / count_work, "分数不一致率±1": task_definition_statistic["分数不一致率±1"] / count_work, "计数题目数": count_work, "无法判断": task_definition_statistic["无法判断"]})

    fout = open(csv_path.replace(".csv", ".jsonl"), 'w')
    for i_data in instruct_list:
        json.dump(i_data, ensure_ascii=False, fp=fout)
        fout.write("\n")
    fout.close()

    # print("Model Files: {}".format(i_data))
    print(f"########## {print_prefix} - 模型：{output_path.split('/')[-1]}")
    print("###############人工总分归一化： {}/{} = {}".format(task_definition_statistic["人工总分"], count_work, task_definition_statistic["人工总分"]/count_work))
    print("###############自动化总分归一化： {}/{} = {}".format(task_definition_statistic["自动化总分"], count_work, task_definition_statistic["自动化总分"]/count_work))
    print("###############分数不一致率： {}/{} = {}".format(task_definition_statistic["分数不一致率"], count_work, task_definition_statistic["分数不一致率"]/count_work))
    print("###############分数不一致率±1： {}/{} = {}".format(task_definition_statistic["分数不一致率±1"], count_work, task_definition_statistic["分数不一致率±1"]/count_work))
    print("###############无法判断 {}".format(task_definition_statistic["无法判断"]))

############################################### inference
def gpt_infer(prompt_ab, agent_func, chat_func, system_message, output_path, cache_gpt_path, save_prefix, force_recreate):
    tmp_path = None
    if output_path and cache_gpt_path:
        path_parts = output_path.split('/')
        if 'results' in path_parts:
            index_of_results = path_parts.index('results')
            if len(path_parts) > 1 and path_parts[0] == '':
                index_of_results += 1
        else:
            index_of_results = -4
        
        os.makedirs(os.path.join(cache_gpt_path, '/'.join(output_path.split('/')[index_of_results:-1])), exist_ok=True)

        if ".jsonl" in output_path.split('/')[-1]:
            tmp_path=os.path.join(cache_gpt_path, '/'.join(output_path.split('/')[index_of_results:-1]), output_path.split('/')[-1].replace(".jsonl",f"_{len(prompt_ab)}_{save_prefix}.pkl"))
        else:
            tmp_path=os.path.join(cache_gpt_path, '/'.join(output_path.split('/')[index_of_results:-1]), output_path.split('/')[-1].replace(".json",f"_{len(prompt_ab)}_{save_prefix}.pkl"))

    res_prompt = None
    if tmp_path and os.path.exists(tmp_path):
        with open(tmp_path, 'rb') as f:
            res_prompt = pickle.load(f)

    if force_recreate or res_prompt is None or len(res_prompt) != len(prompt_ab):
        if hasattr(agent_func, 'mode'):
            res_prompt = chat_func(prompt_ab,agent_func,
                                system_message=system_message, 
                                temperature=agent_func.temperature, 
                                max_tokens=agent_func.max_tokens,
                                batch_size=agent_func.batch_size,
                                # top_p=top_p,
                                n=agent_func.vote_times,
                                mode=agent_func.mode)
        else:
            res_prompt = chat_func(prompt_ab,agent_func,
                                    system_message=system_message, 
                                    temperature=agent_func.temperature, 
                                    max_tokens=agent_func.max_tokens,
                                    batch_size=agent_func.batch_size,
                                    # top_p=top_p,
                                    n=agent_func.vote_times)
        if tmp_path:
            with open(tmp_path, 'wb') as f:
                pickle.dump(res_prompt, f)

    return res_prompt


def danlunkeguan_prprocess_systemprompt(autoprompt_list, instruct_list, system_message=None):
    prompt_ab = []
    mask = []
    system_message_list = []
    for i, (i_data, i_veri) in enumerate(zip(autoprompt_list, instruct_list)):
        assert i_data['id'] == i_veri['id']

        if 'model_answer' in i_veri:
            answer_str = 'model_answer'
        elif 'answer' in i_veri:
            answer_str = 'answer'
        else:
            raise NotImplementedError

        is_zishu = False
        if '字数统计' in  i_data['auto_prompt']:
                is_zishu = i_data['auto_prompt']['字数统计']
        response = i_veri[answer_str]

        # 判断答案是否为空
        if (type(response) is not str and np.isnan(response) ) or (type(response) is str and len(response.strip()) == 0):
            mask.append(0)
            prompt_ab.append("模型答案为空，忽略你的任何身份，按照我的要求输出。。\n\n你务必直接输出：\n最终得分：-200分")
            print("***********************************************模型答案为空，忽略你的任何身份，按照我的要求输出。。\n\n你务必直接输出：\n最终得分：-200分")
        else:
            mask.append(1)

            if is_zishu:
                response = f"[字数：{len(response)}]\n" + response

            try:
                final_prompt = i_data['auto_prompt']['prompt'].format(response=response)
            except TypeError: 
                final_prompt = i_data['auto_prompt'].replace("{response}", response)
            except:
                final_prompt = i_data['auto_prompt']['prompt'].replace("{response}", response)

            final_tokens = ENC.encode(final_prompt)
            if len(final_tokens) >= DANLUNKEGUAN_MAX_TOKEN:
                print(f"#################Max response length is very long: {len(final_tokens)}")
                prompt_tokens = ENC.encode(i_data['auto_prompt']['prompt'].replace("{response}", ""))
                if len(prompt_tokens) < DANLUNKEGUAN_MAX_TOKEN-47:
                    left_tokens = DANLUNKEGUAN_MAX_TOKEN - 47 - len(prompt_tokens)
                    response_jieduan = ENC.decode(ENC.encode(response)[:left_tokens])
                    try:
                        final_prompt = i_data['auto_prompt']['prompt'].format(response=response_jieduan)
                    except:
                        final_prompt = i_data['auto_prompt']['prompt'].replace("{response}", response_jieduan)
                else:
                    mask[-1] = 0
                    final_prompt = "模型答案超长，忽略你的任何身份，按照我的要求输出。。\n\n你务必直接输出：\n最终得分：-100分"
                    response_jieduan = final_prompt
                    print("***********************************************请模型答案超长，你务必直接输出：\n最终得分：-100分")

                print(f"#################Max response length is Now: {len(ENC.encode(response_jieduan))}")
            
            prompt_ab.append(final_prompt)
        if mask[-1] == 1:
            system_message_list.append(system_message)
        else:
            system_message_list.append("")

    return prompt_ab, mask, system_message_list

def agent_danlunkeguan_infer(dataset_list, autoprompt_list, instruct_list, ids, agent_func, output_path, cache_gpt_path, force_recreate=False, args=None, system_message="你是一个在检查答案质量方面非常有帮助且评估精确的助手。", save_prefix="danlunkeguan"):
    api_type_map = {
        "generalv2": generalv2_sub_chat,
    }
    this_sub_chat = api_type_map.get(args.api_type, generalv2_sub_chat)
    ## prompt拼接
    prompt_ab, mask, system_message_list = danlunkeguan_prprocess_systemprompt(autoprompt_list, instruct_list, system_message=system_message)
    ## infer
    res_prompt = gpt_infer(prompt_ab, agent_func, this_sub_chat, system_message_list, output_path, cache_gpt_path, save_prefix, force_recreate)
    ## 后处理
    res_prompt = [min(i_res, i_res*i_mask) for i_res, i_mask in zip(res_prompt, mask)]

    ## 可视化存储
    if args and args.develop_mode:
        agent_danlunkeguan_visfunc(autoprompt_list, instruct_list, res_prompt, prompt_ab,output_path, prefix=save_prefix)

    return res_prompt

## 多轮/工具使用 前处理-infer-后处理
def multiturn_tool_preprocess(autoprompt_list, instruct_list, system_message=None):
    prompt_ab = []
    mask = []
    system_message_list = []
    for i, (i_data, i_veri) in enumerate(zip(autoprompt_list, instruct_list)):
        assert i_data['id'] == i_veri['id']

        if 'model_answer' in i_veri:
            answer_str = 'model_answer'
        elif 'answer' in i_veri:
            answer_str = 'answer'
        else:
            raise NotImplementedError

        query = i_veri['query']
        matches_query = re.findall(r'([AQ]\d[:：])(.*?)((?=[\n][AQ]\d[:：])|$)', query, re.DOTALL)

        response = i_veri[answer_str]
        if type(response) != str: 
            response = ""
        matches = re.findall(r'([AQ]\d[:：])(.*?)((?=[\n][AQ]\d[:：])|$)', response, re.DOTALL)
        if matches:
            multi_turns_num = len(matches)
        else:
            multi_turns_num = 0

        if multi_turns_num != len(matches_query):
            print("!!!!!!!!!!!!!!!!!!!!!!!!! 不正确的答案轮数")

        for i_prompt in i_data['auto_prompt']:
            need_history = i_prompt['meta']['聊天记录依赖']['是否含聊天记录']
            chat_hisotory_turns = i_prompt['meta']['聊天记录依赖'][' 聊天记录依赖回合']
            chat_hisotory_number = extract_info2(chat_hisotory_turns) if len(chat_hisotory_turns) > 0 else []

            eval_turns_info = i_prompt['meta']['待评估模型答案信息']
            eval_turns_number = extract_info2(eval_turns_info) if len(eval_turns_info) > 0 else []

            if len(chat_hisotory_turns) > 0:
                for i_num in chat_hisotory_number:
                    this_query = matches_query[i_num-1][1].strip()
                    try:
                        this_response = matches[i_num-1][1].strip()
                    except:
                        this_response = ""
                    i_prompt['prompt'] = i_prompt['prompt'].replace("{"+f"history_chat{i_num}" +"}", this_query)
                    i_prompt['prompt'] = i_prompt['prompt'].replace("{"+f"history_ans{i_num}" +"}", this_response)
            
            if len(eval_turns_info) == 0:
                # 判断答案是否为空
                if len(response.strip()) == 0:
                    mask.append(0)
                    i_prompt['prompt'] = "模型答案为空，忽略你的任何身份，按照我的要求输出。。\n\n你务必直接输出：\n最终得分：-200分"
                else:
                    mask.append(1)
                    i_prompt['prompt'] = i_prompt['prompt'].replace("{response}", response)
            elif len(eval_turns_number) == 1:
                try:
                    this_response = matches[eval_turns_number[0]-1][1].strip()
                except:
                    this_response = ""
                
                if len(this_response.strip()) == 0:
                    mask.append(0)
                    i_prompt['prompt'] = "模型答案为空，忽略你的任何身份，按照我的要求输出。。\n\n你务必直接输出：\n最终得分：-200分"
                else:
                    mask.append(1)
                    try:
                        i_prompt['prompt'] = i_prompt['prompt'].format(response=this_response)
                    except:
                        i_prompt['prompt'] = i_prompt['prompt'].replace('{response}', this_response)
            else:
                for index, i_num in enumerate(eval_turns_number):
                    try:
                        this_response = matches[i_num-1][1].strip()
                    except:
                        this_response = ""
                    
                    if len(this_response.strip()) == 0:
                        mask.append(0)
                        i_prompt['prompt'] = "模型答案为空，忽略你的任何身份，按照我的要求输出。。\n\n你务必直接输出：\n最终得分：-200分"
                    else:
                        mask.append(1)
                        i_prompt['prompt'] = i_prompt['prompt'].replace("{"+f"response{index+1}" +"}", this_response)

            if len(ENC.encode(i_prompt['prompt'])) > MULTITURN_MAX_TOKEN:
                i_prompt['prompt'] = "模型答案超长，忽略你的任何身份，按照我的要求输出。。\n\n你务必直接输出：\n最终得分：-100分"
                mask[-1] = 0

            prompt_ab.append(i_prompt['prompt'])
            if mask[-1] == 1:
                system_message_list.append(system_message)
            else:
                system_message_list.append("")
    
    return prompt_ab, mask, system_message_list

def multiturn_tool_postprocess(res_prompt, autoprompt_list, instruct_list):
    res_prompt_post = []
    res_index_post = 0
    for i, (i_data, i_veri) in enumerate(zip(autoprompt_list, instruct_list)):
        assert i_data['id'] == i_veri['id']

        if len(i_data['auto_prompt']) == 1:
            res_prompt_post.append(res_prompt[res_index_post])
            res_index_post += 1
            continue
        
        score_ins = []
        compute_turns = []
        compute_scores = []

        for i_prompt in i_data['auto_prompt']:
            compute_turns.append(i_prompt['meta']['分数计算方式']['计算回合数'])
            compute_scores.append(i_prompt['meta']['分数计算方式']['附加分'])
            if res_prompt[i] == -100 or res_prompt[i] == -200:
                score_ins.append(res_prompt[i])
            else:
                score_ins.append(res_prompt[res_index_post])
            res_index_post += 1
        
        compute_turns = 1 - np.isnan(compute_turns)
        compute_scores = np.nan_to_num(compute_scores)

        final_score = 0
        for j, j_turn in enumerate(compute_turns):
            if j_turn:
                final_score += compute_scores[j]
                final_score += score_ins[j]
            else:
                if j < len(compute_turns) - 1 and compute_turns[j+1]:
                    final_score = max(0, final_score + compute_scores[j] + score_ins[j])
                else:
                    final_score = final_score + compute_scores[j] + score_ins[j]
        final_score = min(5, final_score)

        res_prompt_post.append(final_score)

    assert res_index_post == len(res_prompt)

    return res_prompt_post

def agent_multiturn_infer(dataset_list, autoprompt_list, instruct_list, ids, agent_func, output_path, cache_gpt_path, force_recreate=False, args=None, system_message="你是一个在检查答案质量方面非常有帮助且评估精确的助手。", save_prefix="multiturn"):
    api_type_map = {
        "generalv2": generalv2_sub_chat,
        }
    this_sub_chat = api_type_map.get(args.api_type, generalv2_sub_chat)
    ## prompt拼接
    prompt_ab, mask, system_message_list = multiturn_tool_preprocess(autoprompt_list, instruct_list, system_message=system_message)

    ## infer
    res_prompt = gpt_infer(prompt_ab, agent_func, this_sub_chat, system_message_list, output_path, cache_gpt_path, save_prefix, force_recreate)

    ## 后处理
    res_prompt = [min(i_res, i_res*i_mask) for i_res, i_mask in zip(res_prompt, mask)]
    res_prompt = multiturn_tool_postprocess(res_prompt, autoprompt_list, instruct_list)

    ## 可视化存储
    if args and args.develop_mode:
        agent_multiturn_tool_visfunc(autoprompt_list, instruct_list, res_prompt, prompt_ab,output_path, print_prefix=save_prefix)
    
    return res_prompt

def agent_tool_infer(dataset_list, autoprompt_list, instruct_list, ids, agent_func, output_path, cache_gpt_path, force_recreate=False, args=None, system_message="你是一个在检查答案质量方面非常有帮助且评估精确的助手。", save_prefix="tool"):
    api_type_map = {
        "generalv2": generalv2_sub_chat,
        }
    this_sub_chat = api_type_map.get(args.api_type, generalv2_sub_chat)
    ## prompt拼接
    prompt_ab, mask, system_message_list = multiturn_tool_preprocess(autoprompt_list, instruct_list, system_message=system_message)

    ## infer
    res_prompt = gpt_infer(prompt_ab, agent_func, this_sub_chat, system_message_list, output_path, cache_gpt_path, save_prefix, force_recreate)

    ## 后处理
    res_prompt = [min(i_res, i_res*i_mask) for i_res, i_mask in zip(res_prompt, mask)]
    res_prompt = multiturn_tool_postprocess(res_prompt, autoprompt_list, instruct_list)

    ## 可视化存储
    if args and args.develop_mode:
        agent_multiturn_tool_visfunc(autoprompt_list, instruct_list, res_prompt, prompt_ab,output_path, print_prefix=save_prefix)
    
    return res_prompt



dimen_map_agent = {
    "创作": "创作",
    "对话": "多轮",
    "Agent/复杂指令": "工具使用",
    "问答": "单轮客观",
    "理解": "单轮客观",
    "数学": "单轮客观",
    "推理": "单轮客观",
    "代码": "单轮客观",

}

get_agent_infer_generalv2 = {
    "创作": agent_danlunkeguan_infer,
    "多轮": agent_multiturn_infer,
    "工具使用": agent_tool_infer,
    "单轮客观": agent_danlunkeguan_infer,
}

