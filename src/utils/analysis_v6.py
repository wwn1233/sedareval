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
import pandas as pd
from utils import read_json, get_fuse_res_dict, keys_map_fromatscore

#固定种子
np.random.seed(3407)
random.seed(3407)


BASE_PATH=f"./results/"

PREFIX=None #  #"对话,数学-推理-代码-问答-理解"  #"对话.jsonl,数学-推理-代码-问答-理解.jsonl"
SCALE=100  # 5 or 100  代表几分制

DIMENSION_GT={'创作': 48, '对话': 83, '问答': 216, '理解': 203, '推理': 229, '数学': 129, '代码': 31, 'Agent/复杂指令': 61} # sedareval_public

OUTPUT=None # os.path.join(BASE_PATH, BASE_PATH.split("/")[-1] + "_statistics")

KEYS=["总分", "高质量评测题目总分", "问答", "推理", "数学", "理解", "代码", "创作", "对话", "总分归一化", "高质量评测题目数量", "无法判断"]

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', required=False, default=BASE_PATH)
parser.add_argument('--prefix', required=False, default=PREFIX)
parser.add_argument('--scale', required=False, default=SCALE)
parser.add_argument('--dimen_gt', required=False, default=DIMENSION_GT)
parser.add_argument('--outout_path', required=False, help='', default=OUTPUT)

args = parser.parse_args()
# base_path = args.base_path
base_path_list = [i_prefix for i_prefix in args.base_path.split(',')] 
out_put_path = args.outout_path
if out_put_path is None:
    out_put_path = os.path.join(base_path_list[0], "-".join([i_path.split('/')[-1] for i_path in base_path_list]) + "_statistics")
prefix = [i_prefix.split("-") for i_prefix in args.prefix.split(',')] if args.prefix is not None else None
if not os.path.exists(out_put_path):
    os.makedirs(out_put_path, exist_ok=True)
scale = args.scale
dimen_gt = args.dimen_gt


model_info_dict = {}
for base_path in base_path_list:
    model_files = os.listdir(base_path)
    for i_model in model_files:
        if i_model.startswith("analysis") or i_model.startswith("statistics") or i_model.endswith("statistics"):
            continue
        ##判断是文件还是文件夹
        if os.path.isdir(os.path.join(base_path, i_model, "infer_results")): # ***********
            i_model_res = os.listdir(os.path.join(base_path, i_model, "infer_results"))
            i_model_res_list = []
            i_model_res_list_number = []

            if prefix is not None:
                for i_prefix in prefix:
                    if len(i_prefix) == 1 and i_prefix[0] == "":
                        continue
                    i_model_res_list_tmp = []
                    i_model_res_list_number_tmp = []
                    for j_model in i_model_res:
                        if j_model == "origin" or j_model == "analysis":
                            continue
                        components = j_model.split(".")[0].split("-")[1:]
                        if len(components) == len(i_prefix) and set(components) == set(i_prefix):
                            i_model_res_list_tmp.append(j_model)
                            i_model_res_list_number_tmp.append(int(j_model.split("-")[0]))
                    max_index = i_model_res_list_number_tmp.index(max(i_model_res_list_number_tmp))
                    choose_model = i_model_res_list_tmp[max_index]
                    i_model_res_list.append(choose_model)
            else:
                # i_model_res_list_tmp = []
                # i_model_res_list_number_tmp = []
                components=[]
                components_num=[]
                components_index=[]
                for index_component, j_model in enumerate(i_model_res):
                    if j_model == "origin" or j_model == "analysis" or j_model == '.ipynb_checkpoints':
                        continue
                    components.append(set(j_model.split(".")[0].split("-")[1:]))
                    components_num.append(int(j_model.split(".")[0].split("-")[0]))
                    components_index.append(index_component)
                unique_components = list(set([tuple(data) for data in components]))
                
                max_importance_dict = {comp: (-1, -1) for comp in unique_components}
                # 遍历components和components_num
                for i, (comp, num, index) in enumerate(zip(components, components_num, components_index)):
                    # 如果当前元素的重要性大于已存储的最大重要性，则更新字典
                    if num > max_importance_dict[tuple(comp)][0]:
                        max_importance_dict[tuple(comp)] = (num, index)

                # 返回每个元素的最大重要性的索引
                for comp in unique_components:
                    i_model_res_list.append(i_model_res[max_importance_dict[tuple(comp)][1]])

            
            # ##找出i_model_res_list_number中的最大值的index
            # max_index = i_model_res_list_number.index(max(i_model_res_list_number))
            # choose_model = i_model_res_list[max_index]
            model_info_dict[i_model] = {}
            dict_list = []
            for j_model_json in i_model_res_list:
                model_path = os.path.join(base_path, i_model, "infer_results", j_model_json)
                instruct_list = [json.loads(data) for data in read_json(model_path)][0]
                dict_list.append(instruct_list)

            model_info_dict[i_model] = get_fuse_res_dict(dict_list)
        else:
            if not i_model.endswith('.jsonl'):
                continue
            model_path = os.path.join(base_path, i_model)
            instruct_list = [json.loads(data) for data in read_json(model_path)][0]
            model_info_dict[i_model] = instruct_list

model_names_list = list(model_info_dict.keys())



### 总分
keys_list_need = ("问答", "数学", "推理", "理解", "代码", "对话", "Agent/复杂指令", "创作")
all_score_gt_dict = keys_map_fromatscore(keys_list_need, model_info_dict, out_put_path, scale=scale, dimen_gt=dimen_gt, name="总分")


## 单轮客观 
keys_list_need = ("问答", "推理", "数学", "理解", "代码", "Agent/复杂指令") # #("问答", "推理", "理解") 
danlunkeguan_score_gt_dict = keys_map_fromatscore(keys_list_need, model_info_dict, out_put_path, scale=scale, dimen_gt=dimen_gt, name="单轮客观")

# 推理-综
keys_list_need = ("推理", "数学", "代码")
tuilizong_score_gt_dict = keys_map_fromatscore(keys_list_need, model_info_dict, out_put_path, scale=scale, dimen_gt=dimen_gt, name="推理-综")

######################################################################################################
#修正all_score_gt_dict中的自动化均分，推理-综的自动化均分占比为5/6，其他的自动化均分占比为1/6
#Other  ("问答", "理解", "对话", "Agent/复杂指令", "创作")
keys_list_need =  ("问答", "理解", "对话", "Agent/复杂指令", "创作")
other_score_gt_dict = keys_map_fromatscore(keys_list_need, model_info_dict, out_put_path, scale=scale, dimen_gt=dimen_gt, name="其他(问答-理解-对话-Agent_复杂指令-创作)")
for i_model in model_names_list:
    all_score_gt_dict[i_model]["自动化均分"] = (tuilizong_score_gt_dict[i_model]["自动化均分"] + other_score_gt_dict[i_model]["自动化均分"]*5) / 6


keys_list_need = ("问答", "理解", "Agent/复杂指令") # #("问答", "推理", "理解") #
other2_score_gt_dict = keys_map_fromatscore(keys_list_need, model_info_dict, out_put_path, scale=scale, dimen_gt=dimen_gt, name="其他(问答-理解-Agent_复杂指令)")
for i_model in model_names_list:
    danlunkeguan_score_gt_dict[i_model]["自动化均分"] = (tuilizong_score_gt_dict[i_model]["自动化均分"] + other2_score_gt_dict[i_model]["自动化均分"]*3) / 4
########################################################################################################  

## 问答
keys_list_need = ("问答",)
wenda_score_gt_dict = keys_map_fromatscore(keys_list_need, model_info_dict, out_put_path, scale=scale, dimen_gt=dimen_gt, name="问答")

## 数学
keys_list_need = ("数学",)
shuxue_score_gt_dict = keys_map_fromatscore(keys_list_need, model_info_dict, out_put_path, scale=scale, dimen_gt=dimen_gt, name="数学")


## 推理
keys_list_need = ("推理",)
tuili_score_gt_dict = keys_map_fromatscore(keys_list_need, model_info_dict, out_put_path, scale=scale, dimen_gt=dimen_gt, name="推理")

## 理解
keys_list_need = ("理解",)
lijie_score_gt_dict = keys_map_fromatscore(keys_list_need, model_info_dict, out_put_path, scale=scale, dimen_gt=dimen_gt, name="理解")

## 代码
keys_list_need = ("代码",)
daima_score_gt_dict = keys_map_fromatscore(keys_list_need, model_info_dict, out_put_path, scale=scale, dimen_gt=dimen_gt, name="代码")

## 对话
keys_list_need = ("对话",)
duihua_score_gt_dict = keys_map_fromatscore(keys_list_need, model_info_dict, out_put_path, scale=scale, dimen_gt=dimen_gt, name="对话")

## agent
keys_list_need = ("Agent/复杂指令",)
agent_score_gt_dict = keys_map_fromatscore(keys_list_need, model_info_dict, out_put_path, scale=scale, dimen_gt=dimen_gt, name="agent")

## 创作
keys_list_need = ("创作",)
chuangzuo_score_gt_dict = keys_map_fromatscore(keys_list_need, model_info_dict, out_put_path, scale=scale, dimen_gt=dimen_gt, name="创作")


##存储总表格
all_keys = ("all", "danlunkeguan", "lijie", "wenda", "tuilizong", "duihua", "agent", "chuangzuo", "shuxue", "daima", "tuili")
csv_data = [["模型", "模型规模", "是否预训练", "是否自研", "split", "内部-自动化-总分", "战略-人工-总分","内部-自动化-单轮客观","战略-人工-单轮客观","内部-自动化-理解","战略-人工-理解","内部-自动化-问答","战略-人工-问答","内部-自动化-推理_综合","战略-人工-推理_综合","内部-自动化-对话","战略-人工-对话","内部-自动化-复杂指令/agent","战略-人工-复杂指令/agent","内部-自动化-创作","战略-人工-创作", "内部-自动化-数学","战略-人工-数学","内部-自动化-代码","战略-人工-代码", "内部-自动化-推理","战略-人工-推理", "答案空缺率", "无法判断的题目数"]]
INDEX_NUM= 5

### 
for i_model in model_names_list:
    this_data = [""] * len(csv_data[0])
    if i_model.endswith("jsonl"):
        this_data[0] = i_model[:-6]
    else:
        this_data[0] = i_model
    this_data[4] = base_path.split("/")[-1]

    beizhu_info = ""
    for i, i_key in enumerate(all_keys):
        i_key_func = i_key + "_score_gt_dict"
        if i_key_func in locals():
            # score_gt = locals()[i_key_func][i_model]["人工均分"]
            score_auto = locals()[i_key_func][i_model]["自动化均分"]
            # score_gt_num = locals()[i_key_func][i_model]["人工计数"]
            score_auto_num = locals()[i_key_func][i_model]['meta']["自动化计数"]
        else:
            score_gt = "-"
            score_auto = "-"
            score_gt_num = 0
            score_auto_num = 0
        
        index_this = INDEX_NUM + i * 2
        
        this_data[index_this] = score_auto
        
        all_number =  locals()[i_key_func][i_model]['meta']['总题目数']
        if i==0:
            this_data[-2] =  (all_number - score_auto_num) / all_number
        
        beizhu_info += f"  {i_key}：{all_number - score_auto_num}     \n"

    this_data[-1] =  beizhu_info

    csv_data.append(this_data)

##
all_csv_output_path = os.path.join(out_put_path, "总表格.csv")
with open(all_csv_output_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)
    