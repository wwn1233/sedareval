import sys
import os
import json
import argparse
import hashlib
import numpy as np
import random
import csv
import yaml
import pickle
import re
import math

from utils.utils import read_json, get_dimen_map_ids, id_to_agent, get_dimen,get_id_map_exid, get_origin, get_origin_map_ids, get_origin_map_dimens, get_id_map_dimen

from utils.get_judge_agent import dimen_map_agent, get_agent_infer_generalv2
from utils.get_judge_agent_generalv2 import get_all_agents_generalv2

#固定种子
np.random.seed(3407)
random.seed(3407)

CONFIG_BASE=os.path.dirname(os.path.abspath(__file__))

# # 当直接使用main.py 不用argparse从外传递参数时， 评测V3.0时，需要将下面的参数设置为 rubric
DIMENSIONS_PREFIX=None  #   "rubric" #
API_TYPE = "generalv2" 
BASE_PATH="./results"


DATASEET_TEST_SET="./dataset/sedareval_data_ch.json" 
DATASET_TEST_AUTOPROMPT="./dataset/sedareval_data_ch.json" 
EVAL_DOMENSIONS=None   #"问答,理解,数学,推理,代码"  #对话,Agent/复杂指令,创作  # None for ALL
EVAL_ID=None #"025ea3ea051e2dc422e6bf0c887c36ec"
API_KEY=None
EVAL_JSONLS="./dataset/test.jsonl"
OUTPUT_PREFIX="./results/"


EVAL_ID=None #"025ea3ea051e2dc422e6bf0c887c36ec"
API_KEY=None
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"cache")

def str2none(value):
    if value == 'None':
        return None
    return value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_test_set', required=False, help='', default=DATASEET_TEST_SET)  
    parser.add_argument('--dataset_test_aotoprompt', required=False, help='', default=DATASET_TEST_AUTOPROMPT) 
    parser.add_argument('--eval_dimensions', required=False, help='', type=str2none, default=EVAL_DOMENSIONS)  
    parser.add_argument('--dimensions_prefix', required=False, help='', type=str2none, default=DIMENSIONS_PREFIX)  
    parser.add_argument('--api_type', required=False, help='', default=API_TYPE)
    parser.add_argument('--eval_ids', required=False, type=str2none, default=EVAL_ID) 
    parser.add_argument('--eval_jsonls', required=False, help='', default=EVAL_JSONLS)

    parser.add_argument('--base_path', required=False, default=BASE_PATH)
    parser.add_argument('--outout_prefix', required=False, help='', default=OUTPUT_PREFIX)
    parser.add_argument('--api_key', required=False, help='', default=API_KEY)
    parser.add_argument('--cache_gpt_path', required=False, help='用于gpt生成的结果缓存', default=CACHE)
    parser.add_argument('--force_recreate', action="store_true")
    parser.add_argument('--develop_mode', action="store_true") # 存储每个能力维度下详细的结果信息
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()

    print(args)
    dataset_test_set = args.dataset_test_set
    dataset_test_aotoprompt = args.dataset_test_aotoprompt
    eval_dimensions = [i_part for i_part in args.eval_dimensions.split(',') if len(i_part) > 0] if args.eval_dimensions is not None else None
    dimensions_prefix = args.dimensions_prefix
    eval_ids = [i_num for i_num in args.eval_ids.split(',') if len(i_num) > 0] if args.eval_ids is not None else None
    eval_jsonls = [i_jsonl for i_jsonl in args.eval_jsonls.split(',') if len(i_jsonl) > 0] if args.eval_jsonls is not None else None
    base_path = args.base_path
    outout_prefix = args.outout_prefix
    cache_gpt_path = args.cache_gpt_path
    api_key = args.api_key
    api_type = args.api_type
    force_recreate = args.force_recreate

    develop_mode = args.develop_mode
    if not os.path.exists(cache_gpt_path):
        os.makedirs(cache_gpt_path)

    #debug path
    if args.debug:
        outout_prefix = outout_prefix + "/debug"
    args.develop_mode = True

    force_recreate= True

    ## get all gpt4 agent
    print("初始化 GPT4 agents")
    agent_dict_func = {
        "generalv2": get_all_agents_generalv2,
    }
    agent_dict_func = agent_dict_func.get(api_type, "get_all_agents_generalv2")
    agent_dict = agent_dict_func()

    infer_dict_func = {
        "generalv2": get_agent_infer_generalv2,
    }
    infer_func = infer_dict_func.get(api_type, get_agent_infer_generalv2)
    

    ## read  dataset
    print("初始化 测试集 和 自动化prompt")
    tmp_dataset_list = [json.loads(data) for data in read_json(dataset_test_set)]
    tmp_autoprompt_list = [json.loads(data) for data in read_json(dataset_test_aotoprompt)]

    dataset_list = []
    autoprompt_list = []
    for i_data, i_aotoprompt in zip(tmp_dataset_list, tmp_autoprompt_list):
        if 'meta' in i_aotoprompt:
            assert i_data['meta']['id_ex'] == i_aotoprompt['meta']['id_ex']
            if "is_complete" in i_aotoprompt['meta']:
                if i_aotoprompt['meta']['is_complete'] < 0:
                    continue
        dataset_list.append(i_data)
        autoprompt_list.append(i_aotoprompt)
                
    dimen_map_ids = get_dimen_map_ids(dataset_list)
    # dimen_map_agent_expand = get_dimen_map_agent(dataset_list, dimen_map_agent)
    origin_map_ids = get_origin_map_ids(dataset_list)
    origin_map_dimens = get_origin_map_dimens(dataset_list)

    I_dimen, II_dimen, III_dimen = get_dimen(dataset_list)
    id_map_exid = get_id_map_exid(dataset_list)
    id_map_dimen = get_id_map_dimen(dataset_list)
    origin = get_origin(dataset_list)


    ## get eval list , 优先级 ids参数  > 然后按照能力维度参数提取ids
    print("获取评估 id 列表")
    if eval_ids is None:
        if eval_dimensions is None:
            eval_ids = [i_item['id'] for i_item in dataset_list]
        else:
            eval_ids = []
            for i_dimen in eval_dimensions:
                eval_ids.extend(dimen_map_ids[i_dimen])
    agent_ids = []
    for i_id in eval_ids:
        agent_ids.append(dimen_map_agent[id_to_agent(i_id, dataset_list, prefix = dimensions_prefix)])

    #### 将id按照agent类别分组，比如单轮客观、创作、对话、工具调用
    id_groups = {}
    for id, agent in zip(eval_ids, agent_ids):
        if agent not in id_groups:
            id_groups[agent] = []
        id_groups[agent].append(id)

    print("****评测题目数量预估：{}".format(len(eval_ids)))
    eval_dimen_map_num = {}
    for i_id in eval_ids:
        for i_dimen in I_dimen:
            i_ids = dimen_map_ids[i_dimen]
            if i_id in i_ids:
                if i_dimen not in eval_dimen_map_num:
                    eval_dimen_map_num[i_dimen] = 1
                else:
                    eval_dimen_map_num[i_dimen] += 1
    for i_dimen,i_num in eval_dimen_map_num.items():
        print("********{} 维度上评估题目：{}".format(i_dimen, i_num))


    ## start judging
    print("开始评估 ...")
    for i_model in eval_jsonls:
        print("****Processing {}".format(i_model.split('/')[-1][:-6]))

        os.makedirs(outout_prefix, exist_ok=True)
        output_path = os.path.join(outout_prefix, i_model.split('/')[-1])
        output_folder = os.path.join(outout_prefix, i_model.split('/')[-1][:-6])
        json_path = i_model

    instruct_list = [json.loads(data) for data in read_json(json_path)]
    #### 按照agent类型分别计算每一部分的得分
    id_map_score = {}
    for i_agent, i_ids in id_groups.items():
        if len(i_ids) == 0:
            continue
        sub_dataset_list = []
        sub_autoprompt_list = []
        sub_instruct_list = []

        activate_ids = []
        for i_index, i_id in enumerate(i_ids):
            flag=False
            for i_data in instruct_list:
                if i_id == i_data['id'] or ('id_ex' in i_data['meta'] and id_map_exid[i_id] == i_data['meta']['id_ex']):
                    flag=True
                    sub_instruct_list.append(i_data)
                    if i_id != i_data['id']:
                        i_data['id'] = i_id
                    
            if flag:
                for i_data in dataset_list:
                    if i_id == i_data['id']:
                        sub_dataset_list.append(i_data)
                for i_data in autoprompt_list:
                    if i_id == i_data['id']:
                        sub_autoprompt_list.append(i_data)
                activate_ids.append(i_id)
                # if len(sub_dataset_list) != len(sub_instruct_list):
                #     print()
            else:
                # raise ValueError(f"{i_id} not in your files {i_model}")
                print(f"{i_id} not in your files {i_model}")
            
        if len(activate_ids) < len(i_ids):
            print(f"****{i_model} 有{len(i_ids) - len(activate_ids)}个id没有在题集的jsonl文件中找到")

        assert len(sub_dataset_list) == len(sub_autoprompt_list) and len(sub_dataset_list) == len(sub_instruct_list)

        this_score = infer_func[i_agent](sub_dataset_list, sub_autoprompt_list, sub_instruct_list, i_ids, agent_dict[i_agent](api_key), output_path, cache_gpt_path, force_recreate=force_recreate, args=args, system_message="你是一个在检查答案质量方面非常有帮助且评估精确的助手。")
        # for i_index, i_id in enumerate(i_ids):
        for i_index, i_id in enumerate(activate_ids):
            if i_id in id_map_score: ## 有且只有一个分数id
                raise ValueError("Error: 重复的id：{}".format(i_id))
            id_map_score[i_id] = this_score[i_index]
        
        #### 获取人工评分的结果，用作后续对比
        id_map_gtscore = {}
        for i_data in instruct_list:
            if i_data['id'] in id_map_gtscore:
                raise ValueError("重复的id：{}".format(i_data['id']))
            if 'gt' in i_data:
                if type(i_data['gt']) != str and not math.isnan(i_data['gt']):
                    id_map_gtscore[i_data['id']] = i_data['gt']
                else:
                    id_map_gtscore[i_data['id']] = -1
            else:
                id_map_gtscore[i_data['id']] = -1

        #### 后处理，最终成果展示
        ######## 总分  分能力维度  无法判断
        final_score = 0
        error_case = 0
        count = 0
        dimen_map_score = {}
        # count_high_quality = 0
        # high_quality_score = 0

        final_score_gt = 0
        error_case_gt = 0
        count_gt = 0
        dimen_map_score_gt = {}
        # count_high_quality_gt = 0
        # high_quality_score_gt = 0

        for id in eval_ids:
            if id in id_map_score and id in id_map_gtscore:
                if id_map_score[id] >= 0:
                    final_score += id_map_score[id]
                    count += 1

                else:
                    error_case += 1
                
                if id_map_gtscore[id] >= 0:
                    final_score_gt += id_map_gtscore[id]
                    count_gt += 1

                else:
                    error_case_gt += 1

        save_dict = {"自动化-总分": final_score, "自动化-总分归一化": final_score/count, "自动化-计数题目": count, "自动化-无法判断": error_case}
        save_dict["人工-总分"] = final_score_gt
        save_dict["人工-总分归一化"] = final_score_gt/count_gt if count_gt > 0 else 0
        save_dict["人工-计数题目"] = count_gt
        save_dict["人工-无法判断"] = error_case_gt

        # 分维度存储信息
        post_prefix = f"{count}"
        for i_dimen in I_dimen:
            dimen_score = 0
            dimen_score_gt = 0
            i_ids = dimen_map_ids[i_dimen]
            
            count_dimen = 0
            count_dimen_gt = 0
            for i_id in i_ids:
                if i_id in list(id_map_score.keys()) and id_map_score[i_id] >= 0:
                    dimen_score += id_map_score[i_id]
                    count_dimen += 1
                if i_id in list(id_map_gtscore.keys()) and id_map_gtscore[i_id] >= 0:
                    dimen_score_gt += id_map_gtscore[i_id]
                    count_dimen_gt += 1

            if count_dimen > 0:
                save_dict[f"自动化-{i_dimen}({count_dimen})"] = dimen_score
                if i_dimen.startswith("Agent/"):
                    post_prefix += "-"
                    post_prefix += "_".join(i_dimen.split('/'))
                    # post_prefix += f"-{}"
                else:
                    post_prefix += f"-{i_dimen}"
            if count_dimen_gt > 0:
                save_dict[f"人工-{i_dimen}({count_dimen_gt})"] = dimen_score_gt

        ##固定维度名字的顺序
        # 使用"-"分割字符串
        split_string = post_prefix.split("-")
        # 移除第一个元素（"221"）
        prefix_count = split_string.pop(0)
        # 按字母顺序排序
        sorted_string = sorted(split_string)
        # 重新组合字符串
        post_prefix = prefix_count + "-" +"-".join(sorted_string)

        os.makedirs(output_folder, exist_ok=True)
        output_path_other = os.path.join(output_folder, f"{post_prefix}.jsonl")
        fout = open(output_path_other, 'w')
        json.dump(save_dict, ensure_ascii=False, fp=fout)
        fout.write("\n")
        fout.close()

        # 分数据集来源存储信息
        for i_origin in origin:
            osave_dict = {}
            osave_dict[f"自动化-{i_origin}-计数"] = 0
            osave_dict[f"自动化-{i_origin}-总分"] = 0
            osave_dict[f"人工-{i_origin}-计数"] = 0
            osave_dict[f"人工-{i_origin}-总分"] = 0
            osave_dict_besides = save_dict.copy()
            del osave_dict_besides["自动化-总分归一化"]
            osave_dict_besides[f"自动化-总分"] = final_score
            origin_score = 0
            origin_score_gt = 0
            i_ids = origin_map_ids[i_origin] #此来源下的所有id
            count_origin = 0
            count_origin_gt = 0

            for i_id in i_ids:
                i_dimen = id_map_dimen[i_id]
                if i_id in list(id_map_score.keys()) and id_map_score[i_id] >= 0:
                    origin_score += id_map_score[i_id]
                    count_origin += 1
                    if f"自动化-{i_dimen}-得分" in osave_dict:
                        osave_dict[f"自动化-{i_dimen}-得分"]  += id_map_score[i_id]
                        osave_dict[f"自动化-{i_dimen}-计数"]  += 1
                    else:
                        osave_dict[f"自动化-{i_dimen}-得分"]  = id_map_score[i_id]
                        osave_dict[f"自动化-{i_dimen}-计数"]  = 1
                if i_id in list(id_map_gtscore.keys()) and id_map_gtscore[i_id] >= 0:
                    origin_score_gt += id_map_gtscore[i_id]
                    count_origin_gt += 1
                    if f"人工-{i_dimen}-得分" in osave_dict:
                        osave_dict[f"人工-{i_dimen}-得分"]  += id_map_gtscore[i_id]
                        osave_dict[f"人工-{i_dimen}-计数"]  += 1
                    else:
                        osave_dict[f"人工-{i_dimen}-得分"]  = id_map_gtscore[i_id]
                        osave_dict[f"人工-{i_dimen}-计数"]  = 1

            if count_origin > 0:
                osave_dict[f"自动化-{i_origin}-计数"] += count_origin
                osave_dict[f"自动化-{i_origin}-总分"] += origin_score
                osave_dict_besides[f"自动化-总分"] -= osave_dict[f"自动化-{i_origin}-总分"]
                osave_dict_besides[f"自动化-计数题目"] -= osave_dict[f"自动化-{i_origin}-计数"]
                for i_dimen in I_dimen:
                    for i_key in osave_dict:
                        if f"自动化-{i_dimen}-得分" == i_key:
                            for j_key in save_dict:
                                if f"自动化-{i_dimen}" in j_key:
                                    match = re.search(r'\d+', j_key)
                                    number = match.group(0)
                                    i_dimen_count = int(number) - osave_dict[f"自动化-{i_dimen}-计数"]
                                    osave_dict_besides[f"自动化-{i_dimen}({i_dimen_count})"] = save_dict[j_key] - osave_dict[i_key]
                                    del osave_dict_besides[j_key]
                                    break
                    #统一文件显示格式for自动化
                    if f"自动化-{i_dimen}-得分" in osave_dict:
                        i_dimen_count = osave_dict[f"自动化-{i_dimen}-计数"]
                        osave_dict[f"自动化-{i_dimen}({i_dimen_count})"] = osave_dict[f"自动化-{i_dimen}-得分"]
                        del osave_dict[f"自动化-{i_dimen}-计数"]
                        del osave_dict[f"自动化-{i_dimen}-得分"]

            if count_origin_gt > 0:
                osave_dict[f"人工-{i_origin}-计数"] += count_origin_gt
                osave_dict[f"人工-{i_origin}-总分"] += origin_score_gt
                osave_dict_besides[f"人工-总分"] -= osave_dict[f"人工-{i_origin}-总分"]
                osave_dict_besides[f"人工-计数题目"] -= osave_dict[f"人工-{i_origin}-计数"]
                for i_dimen in I_dimen:
                    for i_key in osave_dict:
                        if f"人工-{i_dimen}-得分" in i_key:
                            for j_key in save_dict:
                                if f"人工-{i_dimen}" in j_key:
                                    match = re.search(r'\d+', j_key)
                                    number = match.group(0)
                                    i_dimen_count = int(number) - osave_dict[f"人工-{i_dimen}-计数"]
                                    osave_dict_besides[f"人工-{i_dimen}({i_dimen_count})"] = save_dict[j_key] - osave_dict[i_key]
                                    del osave_dict_besides[j_key]
                                    break
                    #统一文件显示格式for人工
                    if f"人工-{i_dimen}-得分" in osave_dict:
                        i_dimen_count = osave_dict[f"人工-{i_dimen}-计数"]
                        osave_dict[f"人工-{i_dimen}({i_dimen_count})"] = osave_dict[f"人工-{i_dimen}-得分"]
                        del osave_dict[f"人工-{i_dimen}-计数"]
                        del osave_dict[f"人工-{i_dimen}-得分"]

            origin_prefix = os.path.join(output_folder, "origin")
            os.makedirs(origin_prefix, exist_ok=True)

            origin_path = os.path.join(origin_prefix, f"{i_origin}.jsonl")
            fout = open(origin_path, 'w')
            json.dump(osave_dict, ensure_ascii=False, fp=fout)
            fout.write("\n")
            fout.close()

            origin_path = os.path.join(origin_prefix, f"总-{i_origin}.jsonl")
            fout = open(origin_path, 'w')
            json.dump(osave_dict_besides, ensure_ascii=False, fp=fout)
            fout.write("\n")
            fout.close()