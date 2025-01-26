import sys
import os
import json
import random
import numpy as np
from tqdm import tqdm
import re
import transformers
import logging
import csv
import pickle
import pandas as pd
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ['logger', 'buffer_read', 'read_json', 'sub_chat']

def init_logger():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(filename)s: %(lineno)d: %(levelname)s: %(message)s",
                                datefmt='%Y-%m-%d %H:%M:%S')    
    handler.setFormatter(formatter)
    logger = logging.getLogger('GPTDistil')
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger

logger = init_logger()

def buffer_read(f,  buffer_size=20):
    """
    iterate on file handler: f, and return a generator with specified buffer size
    """
    buffer = []
    for line in f:
        if line.strip():
            buffer.append(line.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []
    if len(buffer) > 0:
        yield buffer



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

def extract_content(res, mode="task"):
    try:
        if mode == "task" or mode == "domain":
            result = re.findall(r'<(.*?)>', res)[0]
        elif mode == "task-difficult" or mode == "difficult":
            result = re.findall(r'<(.*?)>', res)[0]
        elif mode == "task-format" or mode == "format":
            result = re.findall(r'<(.*?)>', res)
        else:
            print("")
    except:
        return None

    return result

def merge_dicts(dicts_list):
    merged_dict = {}
    for d in dicts_list:
        merged_dict.update(d)
    return merged_dict


def has_repeated_ngram(text, tokenizer, n=32):
    s = set()
    text = tokenizer.encode(text)
    for i in range(len(text) - n):
        seg = ' '.join([str(_) for _ in text[i:i+n]])
        if seg in s:
            return True
        s.add(seg)


def remove_none(data):
    return [d for d in data if d is not None]


def save_csv(csv_path, instruct_list, task_definition_statistic, keys=["id", "query",  "reference", "answer", "is_hallucination"]):

    # 创建一个新的字典列表，不包含'meta'键
    new_list_of_dicts = [{k: v for k, v in d.items() if k != 'meta'} for d in instruct_list]

    # 获取所有的键（除了'meta'）
    extra_keys = set().union(*new_list_of_dicts) - set(keys)
    keys.extend(extra_keys)

    # 打开你的csv文件，准备写入
    with open(csv_path, "w", newline="", encoding='utf_8_sig') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(new_list_of_dicts)



def gpt_infer_and_cache(chat_func, CACHE, output_path, prompt_ab_I, force_recreate, agent, system_message, temperature, max_tokens, batch_size, vote_times, save_prefix='creation_I', outout_prefix=""):
    ## cache
    os.makedirs(os.path.join(CACHE, outout_prefix, output_path.split('/')[-2]), exist_ok=True)
    tmp_path=os.path.join(CACHE, outout_prefix, output_path.split('/')[-2], output_path.split('/')[-1].replace(".jsonl",f"_{len(prompt_ab_I)}_{save_prefix}.pkl"))
    res_prompt = None
    if os.path.exists(tmp_path):
        with open(tmp_path, 'rb') as f:
            res_prompt = pickle.load(f)

    if force_recreate or res_prompt is None or len(res_prompt) != len(prompt_ab_I):
        res_prompt = chat_func(prompt_ab_I,agent,
                                system_message=system_message, 
                                temperature=temperature, 
                                max_tokens=max_tokens,
                                batch_size=batch_size,
                                # top_p=top_p,
                                n=vote_times)

        with open(tmp_path, 'wb') as f:
            pickle.dump(res_prompt, f)
    
    return res_prompt

def check_arrays(res_prompt_low2score, res_prompt_mid2score, res_prompt_high2score):
    # 首先检查三个数组的长度是否相等
    if len(res_prompt_low2score) != len(res_prompt_mid2score) or len(res_prompt_mid2score) != len(res_prompt_high2score):
        return False

    res_consis = [True] * len(res_prompt_low2score)

    # 遍历数组
    for i in range(len(res_prompt_low2score)):
        # 检查是否满足条件
        if res_prompt_low2score[i] == 0:
            if res_prompt_mid2score[i] == 1:
                if res_prompt_mid2score[i] >= res_prompt_high2score[i]:
                    res_consis[i] = False
            if res_prompt_mid2score[i] == 2:
                if res_prompt_high2score[i] < 2:
                    res_consis[i] = False 
        elif res_prompt_low2score[i] == 1:
            if not (res_prompt_mid2score[i] >= 2 and res_prompt_high2score[i] >= 2):
                res_consis[i] = False
        else:
            if not (res_prompt_mid2score[i] >= 2 and res_prompt_high2score[i] >= 2):
                res_consis[i] = False

    return res_consis


## for 多轮
def extract_info1(chat_turns_needed):
    if chat_turns_needed == "/":
        return "", 0
    result = re.findall("[:：](.*)", chat_turns_needed)[0].strip()
    numbers = [int(num) for num in re.findall("\d+", result)]
    assert len(numbers) == 1 or len(numbers) == 2

    if len(numbers) > 1:
        numbers = [num for num in range(int(numbers[0]), int(numbers[1]) + 1)]

    return result, len(numbers)

def extract_info2(result):
    numbers = [int(num) for num in re.findall("\d+", result)]
    assert len(numbers) == 1 or len(numbers) == 2

    if len(numbers) > 1:
        numbers = [num for num in range(int(numbers[0]), int(numbers[1]) + 1)]
    return numbers



def get_origin(dataset_list):
    origin = set()
    for i_data in dataset_list:
        id = i_data["id"]
        try:
            task_type = i_data["meta"]['数据集来源']
        except Exception as e:
            # print(e)
            task_type = "默认"

        if "," in task_type:
            origins = task_type.split(",")
            for i_origin in origins:
                origin.add(i_origin)
        else:
            origin.add(task_type)
    
    return origin
def get_origin_map_ids(dataset_list):
    origin_map_ids = {}
    for i_data in dataset_list:
        id = i_data["id"]
        try:
            task_type = i_data["meta"]['数据集来源']
        except Exception as e:
            # print(e)
            task_type = "默认"

        if "," in task_type:
            origins = task_type.split(",")
            for i_origin in origins:
                if i_origin in origin_map_ids:
                    origin_map_ids[i_origin].append(id)
                else:
                    origin_map_ids[i_origin] = [id]
        else:
            origin = task_type
            if origin in origin_map_ids:
                origin_map_ids[origin].append(id)
            else:
                origin_map_ids[origin] = [id]
    
    return origin_map_ids

def get_origin_map_dimens(dataset_list):
    origin_map_dimens = {}
    for i_data in dataset_list:
        dimen = i_data["meta"]['category'][0]
        try:
            task_type = i_data["meta"]['数据集来源']
        except Exception as e:
            # print(e)
            task_type = "无数据集来源"
        origin = task_type
        if origin in origin_map_dimens:
            origin_map_dimens[origin].append(dimen)
        else:
            origin_map_dimens[origin] = [dimen]
    
    return origin_map_dimens
def get_id_map_dimen(dataset_list):
    id_map_dimen = {}
    for i_data in dataset_list:
        id = i_data["id"]
        dimen = i_data["meta"]['category'][0]
        id_map_dimen[id] = dimen

    return id_map_dimen


def get_dimen(dataset_list):
    I_dimen = set()
    II_dimen = set()
    III_dimen = set()
    for i_data in dataset_list:
        id = i_data["id"]
        try:
            task_type = i_data["meta"]['category']
        except:
            task_type = i_data["meta"]['task_type']
        I_dimen.add(task_type[0])
        II_dimen.add(task_type[1])
        III_dimen.add(task_type[2])
    
    return I_dimen, II_dimen, III_dimen
def get_dimen_map_ids(dataset_list):
    dimen_map_ids = {}
    for i_data in dataset_list:
        id = i_data["id"]
        try:
            task_type = i_data["meta"]['category']
        except:
            task_type = i_data["meta"]['task_type']
        I_dimen, II_dimen, III_dimen = task_type[0], task_type[1], task_type[2]

        if I_dimen in dimen_map_ids:
            dimen_map_ids[I_dimen].append(id)
        else:
            dimen_map_ids[I_dimen] = [id]
    
    return dimen_map_ids

def get_high_quality_ids(dataset_list):
    ids = []
    for i_data in dataset_list:
        id = i_data["id"]

        if "high_quality" in i_data['meta']:
            if i_data['meta']['high_quality']:
                ids.append(id)
    
    return ids

def get_dimen_map_agent(dataset_list, dimen_map_agent):
    dimen_map_agent_expand = {}
    
    for i_data in dataset_list:
        id = i_data["id"]
        try:
            task_type = i_data["meta"]['category']
        except:
            task_type = i_data["meta"]['task_type']
        I_dimen, II_dimen, III_dimen = task_type[0], task_type[1], task_type[2]
        agent_type = dimen_map_agent[I_dimen]

        if I_dimen not in dimen_map_agent_expand:
            dimen_map_agent_expand[I_dimen] = agent_type
        if II_dimen not in dimen_map_agent_expand:
            dimen_map_agent_expand[II_dimen] = agent_type
        if III_dimen not in dimen_map_agent_expand:
            dimen_map_agent_expand[III_dimen] = agent_type

    return dimen_map_agent_expand

def id_to_agent(i_id, dataset_list, prefix=None):
    for i_data in dataset_list:
        if i_data["id"] == i_id:
            try:
                res = i_data["meta"]['category'][0]
            except:
                res = i_data["meta"]['task_type'][0]
            if prefix:
                res = prefix +"_"+ res
            return res
    raise ValueError("No solution")

def get_id_map_exid(dataset_list):
    id_map_exid = {}
    for i_data in dataset_list:
        id = i_data["id"]
        exid = i_data["meta"]["id_ex"]
        id_map_exid[id] = exid

    return id_map_exid

def get_gsb_value(model_names_list, model_score_gt_dict, thresh_gt=3, thresh_auto=3, save_path=""):
    # 生成所有模型的两两组合
    model_combinations = list(combinations(model_names_list, 2))
    if save_path != "":
        save_file = os.path.join(save_path, "GSB_混淆矩阵.csv")
        # save_png_file = os.path.join(save_path, "GSB_混淆矩阵.png")
        # 创建一个空的DataFrame
    df = pd.DataFrame(index=[i for i in model_names_list], columns=[i for i in model_names_list])

    # 对所有模型组合进行比较
    count_combinations = 0
    count_consis = 0
    for model1_name, model2_name in model_combinations:
        # 选择两个模型
        model1_gt = model_score_gt_dict[model1_name]['人工分数']
        model2_gt = model_score_gt_dict[model2_name]['人工分数']
        model1_gt_num = model_score_gt_dict[model1_name]['人工计数']
        model2_gt_num = model_score_gt_dict[model2_name]['人工计数']
        assert model1_gt_num == model2_gt_num

        model1_auto = model_score_gt_dict[model1_name]['自动化分数']
        model2_auto = model_score_gt_dict[model2_name]['自动化分数']
        model1_auto_num = model_score_gt_dict[model1_name]['自动化计数']
        model2_auto_num = model_score_gt_dict[model2_name]['自动化计数']
        
        ## 归一化到一个尺度上，避免计数题目上的差异导致嘴周结果的差异
        if model1_auto_num != model1_gt_num:
            if model1_auto_num == 0:
                model1_auto = 0
            else:
                model1_auto = (model1_auto/model1_auto_num) * model1_gt_num
        if model2_auto_num != model2_gt_num:
            if model2_auto_num == 0:
                model2_auto = 0
            else:
                model2_auto = (model2_auto/model2_auto_num) * model2_gt_num

        
        count_combinations += 1
        # 人工打分模型差异
        gsb_gt=None
        if abs(model1_gt - model2_gt) > thresh_gt:
            if model1_gt > model2_gt:
                gsb_gt = "g"
            else:
                gsb_gt = "b"
        else:
            gsb_gt = "s"
        

        # 自动打分模型差异
        gsb_auto=None
        if abs(model1_auto - model2_auto) > thresh_auto:
            if model1_auto > model2_auto:
                gsb_auto = "g"
            else:
                gsb_auto = "b"
        else:
            gsb_auto = "s"
        
        if gsb_gt == gsb_auto:
            count_consis += 1
            df.loc[model1_name, model2_name] = 1
            df.loc[model2_name, model1_name] = 1
        else:
            df.loc[model1_name, model2_name] = 0
            df.loc[model2_name, model1_name] = 0
        
    if save_path != "":
        # 将NaN值填充为0
        df = df.fillna(1)
        df.to_csv(save_file, encoding='utf-8-sig')
    return count_consis/count_combinations, count_combinations


# {"自动化-总分": "xx", "自动化-总分归一化": "xx", "自动化-计数题目": "xx", "自动化-高质量评测题目总分": "xx", "自动化-高质量评测题目数量": "xx", "自动化-无法判断": "xx", "人工-总分": "xx", "人工-总分归一化": "xx", "人工-计数题目": "xx", "人工-高质量评测题目总分": "xx", "人工-高质量评测题目数量": "xx", "人工-无法判断": "xx", "人工-创作(40)": "xx", "自动化-推理(35)": "xx", "人工-推理(35)": "xx", "自动化-代码(22)": "xx", "人工-代码(22)": "xx", "自动化-问答(70)": "xx", "人工-问答(70)": "xx", "自动化-数学(26)": "xx", "人工-数学(26)": "xx", "自动化-理解(68)": "xx", "人工-理解(68)": "xx", "自动化-对话(38)": "xx", "人工-对话(39)": "xx"}
def keys_map_gsb(keys_list_need, model_info_dict, out_put_path, name="总分", thresh_gt=3, thresh_auto=3):
    model_names_list = list(model_info_dict.keys())

    model_score_gt_dict = {}
    for i_model in model_names_list:
        score_auto = 0
        score_gt = 0
        count_auto = 0
        count_gt = 0
        this_info_dict = model_info_dict[i_model]
        for i_key in keys_list_need:
            for j_key, j_value in this_info_dict.items():
                if f"人工-{i_key}" in j_key:
                    score_gt += j_value
                    if i_key == '高质量评测题目总分':
                        count_gt += this_info_dict['人工-高质量评测题目数量']
                    else:
                        numbers = re.findall('\d+', j_key)
                        count_gt += int(numbers[0])
                elif f"自动化-{i_key}" in j_key:
                    score_auto += j_value
                    if i_key == '高质量评测题目总分':
                        count_auto += this_info_dict['自动化-高质量评测题目数量']
                    else:
                        numbers = re.findall('\d+', j_key)
                        count_auto += int(numbers[0])
                else:
                    continue
        model_score_gt_dict[i_model] = {
            "人工分数": score_gt,
            "自动化分数": score_auto,
            "人工计数": count_gt,
            "自动化计数": count_auto
        }

    gsb_value, count_number = get_gsb_value(model_names_list, model_score_gt_dict, thresh_gt=thresh_gt, thresh_auto=thresh_auto, save_path=out_put_path)

    print(f"{name}，GSB一致率为（{count_number}对）：", gsb_value)

    csv_data = [["模型", "人工分数", "自动化分数", "人工计数", "自动化计数"]]
    for i_model in model_names_list:
        csv_data.append([
            i_model,
            model_score_gt_dict[i_model]["人工分数"],
            model_score_gt_dict[i_model]["自动化分数"],
            model_score_gt_dict[i_model]["人工计数"],
            model_score_gt_dict[i_model]["自动化计数"]
        ])
    csv_data.append([f"{name} - GSB一致率（{count_number}回合）：", gsb_value, "", "", ""])
    ##存储csv文件
    csv_output_file = os.path.join(out_put_path, f"{name}.csv")
    with open(csv_output_file, 'w', newline='', encoding='utf-8-sig') as f:  #, encoding='utf-8'
        writer = csv.writer(f)
        # 写入数据
        for row in csv_data:
            writer.writerow(row)
    ##存储jsonl文件
    jsonl_output_file = os.path.join(out_put_path, f"{name}.jsonl")
    with open(jsonl_output_file, 'w', newline='', encoding='utf-8') as fout:
        json.dump({f"GSB一致率": gsb_value, "回合数": count_number}, ensure_ascii=False, fp=fout)
        fout.write("\n")
        for i_model in model_names_list:
            json.dump({i_model:model_score_gt_dict[i_model]}, ensure_ascii=False, fp=fout)
            fout.write("\n")
    
    return gsb_value, count_number, model_score_gt_dict

### scale分制下的 人工总分/自动化总分    
def keys_map_fromatscore(keys_list_need, model_info_dict, out_put_path, scale=100, dimen_gt=None, name="总分"):
    model_names_list = list(model_info_dict.keys())

    assert dimen_gt is not None
    ##遍历 dimen_gt， 累加
    num_all = 0
    for i_key, i_value in dimen_gt.items():
        if i_key in keys_list_need:
            num_all += i_value


    model_score_gt_dict = {}
    for i_model in model_names_list:
        score_auto = 0
        score_gt = 0
        count_auto = 0
        count_gt = 0
        this_info_dict = model_info_dict[i_model]
        ema_average_gt = 0
        ema_average_auto = 0
        count_keys_gt = 0
        count_keys_auto = 0

        model_score_keys_dict = {}
        for i_key in keys_list_need:
            if i_key == '高质量评测题目总分':
                continue
            
            model_score_keys_dict[i_key] = {}
            model_score_keys_dict[i_key]['题目个数（全部）'] = dimen_gt[i_key]
            for j_key, j_value in this_info_dict.items():
                if f"人工-{i_key}" in j_key:
                    score_gt += j_value

                    numbers = re.findall('\d+', j_key)
                    count_gt += int(numbers[0])

                    score_gt_norm =  float(j_value * scale / (int(numbers[0]) * 5)) if int(numbers[0]) != 0 else 0
                    model_score_keys_dict[i_key].update({"人工分数（有效题目）": score_gt_norm, "人工题目个数（有效）": int(numbers[0])})              

                    ema_average_gt += score_gt_norm
                    count_keys_gt += 1
                elif f"自动化-{i_key}" in j_key:
                    score_auto += j_value
                    
                    numbers = re.findall('\d+', j_key)
                    count_auto += int(numbers[0])

                    if i_key == "代码执行" or i_key == "奥林匹克-数学":
                        score_auto_norm = float(j_value * scale / (int(numbers[0]) )) if int(numbers[0]) != 0 else 0
                    else:
                        score_auto_norm = float(j_value * scale / (int(numbers[0]) * 5)) if int(numbers[0]) != 0 else 0
                    model_score_keys_dict[i_key].update({"自动化分数（有效题目）": score_auto_norm, "自动化题目个数（有效）": int(numbers[0])})

                    ema_average_auto += score_auto_norm
                    count_keys_auto += 1
                else:
                    continue
                    # raise ValueError(f"key {j_key} not Normal")
        model_score_gt_dict[i_model] = {"自动化均分": ema_average_auto/count_keys_auto if count_keys_auto != 0 else 0, "人工均分": ema_average_gt/count_keys_gt if count_keys_gt != 0 else 0}
        model_score_gt_dict[i_model].update({"子维度信息": model_score_keys_dict})
        model_score_gt_dict[i_model].update({
            'meta': {
            "自动化分数": float(score_auto * scale / (count_auto * 5)) if count_auto != 0 else 0,
            "自动化计数": count_auto,
            "人工分数": float(score_gt * scale / (count_gt * 5)) if count_gt != 0 else 0,
            "人工计数": count_gt,
            "分制": scale,
            "总题目数": num_all
            }
        })
        
    csv_data = [["模型", "人工均分", "自动化均分", "人工计数", "自动化计数"]]
    for i_model in model_names_list:
        csv_data.append([
            i_model,
            model_score_gt_dict[i_model]["人工均分"],
            model_score_gt_dict[i_model]["自动化均分"],
            model_score_gt_dict[i_model]['meta']["人工计数"],
            model_score_gt_dict[i_model]['meta']["自动化计数"]
        ])
    ##存储csv文件
    csv_output_file = os.path.join(out_put_path, f"{name}.csv")
    with open(csv_output_file, 'w', newline='', encoding='utf-8-sig') as f:  #, encoding='utf-8'
        writer = csv.writer(f)
        # 写入数据
        for row in csv_data:
            writer.writerow(row)
    ##存储jsonl文件
    jsonl_output_file = os.path.join(out_put_path, f"{name}.jsonl")
    with open(jsonl_output_file, 'w', newline='', encoding='utf-8') as fout:
        for i_model in model_names_list:
            json.dump({i_model:model_score_gt_dict[i_model]}, ensure_ascii=False, fp=fout)
            fout.write("\n")
    
    return model_score_gt_dict


def response_long_check(response, prompt_template, encoder, context_num=8147, default_prompt="请你务必直接输出：\n最终得分：0分"):
    try:
        final_prompt = prompt_template.format(response=response)
    except:
        final_prompt = prompt_template.replace("{response}", response)

    is_mask = False
    final_tokens = encoder.encode(final_prompt)
    if len(final_tokens) >= context_num:
        print(f"#################Max response length is very long: {len(final_tokens)}")
        prompt_tokens = encoder.encode(prompt_template.replace("{response}", ""))
        if len(prompt_tokens) < (context_num-47):
            left_tokens = (context_num-47) - len(prompt_tokens)
            response_jieduan = encoder.decode(encoder.encode(response)[:left_tokens])
            try:
                final_prompt = prompt_template.format(response=response_jieduan)
            except:
                final_prompt = prompt_template.replace("{response}", response_jieduan)
        else:
            is_mask = True
            final_prompt = default_prompt  #"请你务必直接输出：\n最终得分：0分"
            print("***********************************************", default_prompt)

        print(f"#################Max response length is Now: {len(encoder.encode(response_jieduan))}")

    return final_prompt, is_mask

def filter_keys(data):
    # 创建一个空字典来存储结果
    result = {}

    # 遍历集合中的每个元素
    for item in data:
        # 如果元素包含括号
        if '(' in item:
            # 提取元素的前缀和括号中的数字
            prefix, number = item.split('(')
            number = int(number[:-1])
            # 如果前缀已经在结果字典中，并且当前数字大于字典中的数字
            if prefix in result and number > int(result[prefix].split('(')[-1][:-1]):
                # 更新字典中的数字
                result[prefix] = item
            # 如果前缀不在结果字典中
            elif prefix not in result:
                # 将前缀和数字添加到字典中
                result[prefix] = item
        # 如果元素不包含括号
        else:
            # 直接将元素添加到结果字典中，值为None
            result[item] = item
    
    result = list(result.values())

    return result

def get_fuse_res_dict(sub_dict):
    model_info_dict = {}
    all_keys = set()
    for i_dict in sub_dict:
        all_keys = all_keys.union(set(i_dict.keys()))

    all_keys = filter_keys(all_keys)

    # 将A1, A2, A3合并到model_info_dict[i_model]
    for key in all_keys:
        if key in ["自动化-总分", "自动化-计数题目", "自动化-高质量评测题目总分", "自动化-高质量评测题目数量", "自动化-无法判断", "人工-总分", "人工-计数题目", "人工-高质量评测题目总分", "人工-高质量评测题目数量", "人工-无法判断"]:
            model_info_dict[key] = 0
            for i_dict in sub_dict:
                value = i_dict.get(key, 0)
                model_info_dict[key] += value
        elif key in ["自动化-总分归一化", "人工-总分归一化"]:
            model_info_dict[key] = 0
        elif key.startswith("人工"):
            reference = -1000
            flag = False
            for i_dict in sub_dict:
                value = i_dict.get(key, -1000)
                if not flag:
                    if value >= 0:
                        reference = value
                        flag = True
                else:
                    if reference >= 0 and value >= 0:
                        assert value==reference
            if reference >= 0:
                model_info_dict[key] = reference
        else:
            for i_dict in sub_dict:
                value = i_dict.get(key, -1000)
                if value >= 0:
                    model_info_dict[key] = value
                    break

    return model_info_dict