import asyncio
from tqdm.asyncio import tqdm_asyncio
try:
    from openai import OpenAI   #, AsyncOpenAI
except:
    print("Version of openai is too low, please upgrade to 1.0.0 or higher")

from tqdm import tqdm
import re
from .utils import logger
import time
import os
import math
import collections
import threading
# from tqdm.contrib.concurrent import process_map
from concurrent.futures import ThreadPoolExecutor,as_completed
from functools import partial
from collections import Counter

# pip3 install openai aioretry
from aioretry import (
    RetryPolicyStrategy,
    RetryInfo,
    RetryPolicy,
    retry as aretry
)

wait_time_list = [0.1, 0.3, 0.5, 1, 1, 0.1, 0.3, 2, 0.3, 0.3, 2, 2, 0.1, 0.1, 5, 1, 0.5, 0.5, 1, 1, 30, 2, 30] * 4
def retry_policy(info: RetryInfo) -> RetryPolicyStrategy:
        if info.fails % 6 == 0:
            logger.info(f"OPENAI Request Retrying {info.fails} times")

        # return False,wait_time
        if info.fails > len(wait_time_list):
            return True, 120
        
        return False, wait_time_list[info.fails - 1]

from openai.types.chat.chat_completion import ChatCompletion

def merge_chat_completions(completions):
    try:
        # 筛选出没有错误的 ChatCompletion 对象
        valid_completions = [comp for comp in completions if 'error' not in comp]
        
        # 如果没有有效的 ChatCompletion 对象，返回错误或空的 ChatCompletion 对象
        if not valid_completions:
            return completions[0]
        
        # 使用第一个没有错误的 ChatCompletion 对象作为基础
        base_completion = valid_completions[0]
        
        # 遍历剩余的 ChatCompletion 对象，合并 choices
        for completion in valid_completions[1:]:
            base_completion.choices.extend(completion.choices)
        
        # 返回合并后的 ChatCompletion 对象
        return base_completion
    except Exception as e:
        logger.error(f"merge_chat_completions error: {e}")
        return completions[0]


class OpenAIGPT_General_V2:

    def __init__(self, temperature=0.8, max_tokens=256, vote_times=1, top_p=1, batch_size=20, mode=0, **kwargs):
        
        api_key = kwargs.pop("api_key", "demo_api_1")
        api_base = kwargs.pop("api_base", "https://**.com/eval/v1") 

        self.model = kwargs.pop("model", "gpt4o")
        self.pbar = None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.vote_times = vote_times
        self.batch_size = batch_size
        self.mode = mode
        self.best_of = kwargs.pop("best_of", 1)
        self.presence_penalty = kwargs.pop("presence_penalty", 0)

        self.client =  OpenAI(api_key=api_key, base_url=api_base)

        ## info vis
        self.tokens_count = [0, 0, 0]  # total_tokens, prompt_tokens, completion_tokens
        self.total_request = 0
        self.success_request = 0
        self.request_queue = collections.deque()
        self.lock = threading.Lock()

        logger.info(f"OpenAIGPT_General_V2 init with model: {self.model}, temperature: {self.temperature}, max_tokens: {self.max_tokens}, vote_times: {self.vote_times}, top_p: {self.top_p}, batch_size: {self.batch_size}, mode: {self.mode}")
        
    def completion(self, question, system_message, temperature=0.8, max_tokens=256,top_p=1, n=1):
        if isinstance(system_message, list):
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=system_message + [
                        {"role": "user",
                        "content": question}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                )
        else:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                    "content": system_message},
                    {"role": "user",
                    "content": question}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                )
        # print(chat_completion)
        with self.lock:
            self.total_request += 1
            self.request_queue.append((time.time(), False))

        return chat_completion  #['choices'][0]['message']['content']

    @aretry(retry_policy)
    async def async_completion(self, question, system_message, temperature=0.8, max_tokens=256,top_p=1, n=1):
        if isinstance(system_message, list):
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=system_message + [
                        {"role": "user",
                        "content": question}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                )
        else:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                    "content": system_message},
                    {"role": "user",
                    "content": question}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                )
        with self.lock:
            self.total_request += 1
            self.request_queue.append((time.time(), False))

        return chat_completion  #['choices'][0]['message']['content']


    async def get_async_completion(self, question, system_message, temperature=0.8, max_tokens=256,top_p=1, n=1):
        tasks = [self.async_completion(question,
                            system_message=system_message,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p) for _ in range(n)]
        resp_list = await asyncio.gather(*tasks)

        return merge_chat_completions(resp_list)

    async def get_async_completion2(self, questions, system_message, temperature=0.8, max_tokens=256,top_p=1, n=1):
        results = [self.async_completion(q,
                    system_message=sys,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p) for q, sys in zip(questions,system_message) ]
        resp_list = await asyncio.gather(*results)

        return resp_list


    def batch_async_completion(self, questions,
                                     system_message="",
                                     temperature=0.8,
                                     max_tokens=512,
                                     top_p=1,
                                     n=1,
                                     mode=0,
                                     thread_id=0):
        pbar = None
        if type(system_message) == str:
            system_message = [system_message] * len(questions)
        
        if n > 1:
            results = [asyncio.run(self.get_async_completion(q,
                                        system_message=sys,
                                        temperature=temperature,
                                        max_tokens=max_tokens,
                                        top_p=top_p,
                                        n=n)) for q, sys in zip(questions,system_message) ]
            # results = await asyncio.gather(*tasks)
        else:
            results = asyncio.run(self.get_async_completion2(questions,
                                        system_message=system_message,
                                        temperature=temperature,
                                        max_tokens=max_tokens,
                                        top_p=top_p))
                            
        rets = []
        for idx, value in enumerate(results):
            num=0
            this_max_tokens = max_tokens
            while True:
                if 'error' not in value:
                    try:
                        flag = False
                        for choice in value.choices:
                            # logger.info(choice.message.content[0]['text'])
                            match = re.search(r'最终得分为[:：]\s*(-?\d+(?:\.\d+)?)|最终得分[:：]\s*(-?\d+(?:\.\d+)?)', choice.message.content[0]['text'] if "text" in choice.message.content[0] else choice.message.content)
                            if match:
                                flag = True
                                break
                        if flag:
                            ret = value
                            break
                    except:
                        pass
                time.sleep(3)
                ## 处理是否因为输出超长导致的提取结果模式失败
                try:
                    if value['choices'][0]['finishReason'] == 'length':
                        this_max_tokens += 512 # 256
                except:
                    pass
                value = self.completion(questions[idx], system_message=system_message[idx],
                                temperature=temperature,
                                max_tokens=this_max_tokens,
                                top_p=top_p,
                                n=n)

                if num > 50: #200
                    # logger.info(questions)
                    logger.error(f'{value} - {idx} for {questions[idx]}')  # [:700]
                    ret = None
                    # print("WWN-2-1-3-1")
                    break
                num += 1
            
            ## get results
            if ret is not None:
                with self.lock:
                    self.success_request += 1
                    for i in range(len(self.request_queue) - 1, -1, -1):
                        if self.request_queue[i][1] == False:
                            self.request_queue[i] = (self.request_queue[i][0], True)
                            break
                    self.tokens_count[0] += ret.usage.total_tokens
                    self.tokens_count[1] += ret.usage.prompt_tokens
                    self.tokens_count[2] += ret.usage.completion_tokens

                outputs = []
                count_work = 0
                for choice in ret.choices:
                    match = re.findall(r'最终得分为[:：]\s*(-?\d+(?:\.\d+)?)|最终得分[:：]\s*(-?\d+(?:\.\d+)?)', choice.message.content[0]['text'] if "text" in choice.message.content[0] else choice.message.content)
                    if len(match) > 0:
                        count_work += 1
                        # predict_score = float(match.group(1)) if match.group(1) else float(match.group(2))
                        predict_score = float(match[-1][0]) if len(match[-1][0])>0 else float(match[-1][1])

                        if predict_score < 0:
                            outputs = []
                            count_work = 0
                            break

                        outputs.append(predict_score)
                
                if count_work > 0:
                    assert len(outputs) == count_work
                    if mode == 0:
                        ret = sum(outputs) / count_work
                    elif mode == 1:
                        score_dict = {}
                        for i_score in outputs:
                            if i_score in score_dict:
                                score_dict[i_score] += 1
                            else:
                                score_dict[i_score] = 1
                        keys_score = list(score_dict.keys())
                        keys_score.sort()
                        if len(keys_score) == 1:
                            ret = keys_score[0]
                        elif len(keys_score) == 2:
                            ret = keys_score[0]  #*0.9 + keys_score[1]*0.1
                        elif len(keys_score) == 3 or len(keys_score) == 4:
                            ret = keys_score[0] #*0.8 + keys_score[1]*0.2
                        else:
                            # ret = keys_score[0]*0.4 + keys_score[1]*0.2 + keys_score[2]*0.2 + keys_score[3]*0.2
                            weights = [0.4, 0.2, 0.2, 0.2]
                            ret = sum(keys_score[i] * weights[i] for i in range(min(len(keys_score), len(weights))))
                    elif mode == 2:
                        score_dict = {}
                        for i_score in outputs:
                            if i_score in score_dict:
                                score_dict[i_score] += 1
                            else:
                                score_dict[i_score] = 1
                        keys_score = list(score_dict.keys())
                        keys_score.sort(reverse=True)
                        if len(keys_score) == 1:
                            ret = keys_score[0]
                        elif len(keys_score) == 2:
                            ret = keys_score[0]  #*0.9 + keys_score[1]*0.1
                        elif len(keys_score) == 3 or len(keys_score) == 4:
                            ret = keys_score[0] #*0.8 + keys_score[1]*0.2
                        else:
                            # ret = keys_score[0]*0.4 + keys_score[1]*0.2 + keys_score[2]*0.2 + keys_score[3]*0.2
                            weights = [0.4, 0.2, 0.2, 0.2]
                            ret = sum(keys_score[i] * weights[i] for i in range(min(len(keys_score), len(weights))))
                    elif mode == 3:
                        counter = Counter(outputs)
                        ret = counter.most_common(1)[0][0]
                    else:
                        raise ValueError("")

                else:
                    ret = -100
            else:
                ret = -100
            rets.append(ret)
        return rets


    def get_used_tokens(self):
        with self.lock:
            return self.tokens_count[0], self.tokens_count[1], self.tokens_count[2]

    def get_request_num(self):
        with self.lock:
            return self.total_request, self.success_request

    def get_request_num_recent(self):
        now = time.time()
        total_request_recent = 0
        success_request_recent = 0
        with self.lock:
            while self.request_queue and now - self.request_queue[0][0] > 300:  # 存储近5分钟的请求
                self.request_queue.popleft()
            for request in self.request_queue:
                total_request_recent += 1
                if request[1] == True:
                    success_request_recent += 1
        return total_request_recent, success_request_recent
    

def chat(prompt, agent, system_message, temperature, max_tokens, reverse=False,top_p=1, n=1,mode=0,thread_id=0):
    count_num = 0
    while True:
        try:
            count_num += 1

            if not isinstance(prompt, list):
                batch_query = [prompt]
            else:
                batch_query = prompt
                
            rets = agent.batch_async_completion(questions=batch_query,
                                                        system_message=system_message,
                                                        temperature=temperature,
                                                        max_tokens=max_tokens,
                                                        top_p=top_p,
                                                        n=n,
                                                        mode=mode,
                                                        thread_id=thread_id) #[0]

            assert len(prompt) == len(rets), "mismatch for input/output batch size"

            return rets
        except Exception as e:
            # pass
            if count_num > 30:  #100:
                logger.error(f'请求超时，重试次数超过3次，跳过！错误消息见：{e}')
                return [-100] * len(prompt)

            logger.info(f'请求超时，重试中! 错误消息见：{e}')
            time.sleep(3)

def generalv2_sub_chat_single(prompt_ab, agent,system_message, temperature, max_tokens, batch_size=20, reverse=False, top_p=1, n=1,mode=1):
    res = []
    for i in tqdm(range(0, len(prompt_ab), batch_size), desc="整体任务进度：", position=0):
        tqdm.write(' ')
        end = min(i+batch_size, len(prompt_ab))
        chunk = prompt_ab[i:end]
        # print("WWN-2-1")
        res.extend(chat(chunk,agent,system_message=system_message, temperature=temperature, max_tokens=max_tokens,top_p=top_p,n=n,mode=mode))
        # print("WWN-2-2")
    return res


def allocate_threads(S):
    if S <= 5:
        thread_num = 1
        chunk_num = 1
        ins_num = S
    else:
        ins_num = 5
        while S / ins_num > 16:  # or ins_num * (S // ins_num) < S
            ins_num += 1
        thread_num = min(16, math.ceil(S // ins_num))
        chunk_num = thread_num

        while chunk_num * ins_num < S:
            ins_num += 1

    return thread_num, chunk_num, ins_num


def chat_func(chunk, agent, temperature, max_tokens, top_p, n, mode):
    chunk, system_message = chunk
    thread_id = threading.get_ident()
    return chat(chunk, agent, system_message=system_message, temperature=temperature, max_tokens=max_tokens,top_p=top_p,n=n,mode=mode, thread_id=thread_id)

def generalv2_sub_chat(prompt_ab, agent,system_message, temperature, max_tokens, batch_size=20, reverse=False, top_p=1, n=1,mode=1):
    if batch_size > 256:
        print("batch_size should be less than 256!!!")
        batch_size = 256

    res = []

    if type(system_message) == str:
        system_message = [system_message] * len(prompt_ab)
    
    chat_func_partial = partial(chat_func, agent=agent, temperature=temperature, max_tokens=max_tokens, top_p=top_p, n=n, mode=mode)

    pre_chunk_num = max(1, math.ceil(len(prompt_ab) / batch_size))
    ins_num = math.ceil(len(prompt_ab) / pre_chunk_num)
    thread_num, chunk_num, ins_num = allocate_threads(ins_num)
    logger.info(f"整体评测样本分为 {pre_chunk_num} 组（【整体任务进度】），每组最大 {ins_num} 个样本（分片任务整体进度），预计最大 {thread_num} 个线程并行处理。（<每个分片任务进度>）")
    for i in tqdm(range(0, len(prompt_ab), batch_size), desc="####【整体任务进度】####：", position=0):
        end = min(i+batch_size, len(prompt_ab))
        chunk = prompt_ab[i:end]
        sys_chunk = system_message[i:end]

        thread_num, chunk_num, ins_num = allocate_threads(len(chunk))

        sub_chunks = [(chunk[i:min(i+ins_num, len(chunk))], sys_chunk[i:min(i+ins_num, len(sys_chunk))]) for i in range(0, len(chunk), ins_num)]

        sub_res = [None] * len(sub_chunks)
        with ThreadPoolExecutor(max_workers=thread_num) as executor:
            futures = {executor.submit(chat_func_partial, chunk): i for i, chunk in enumerate(sub_chunks)}
            pbar = tqdm(total=len(futures), desc="      ########【分片任务整体进度】########：", position=1)
            for future in as_completed(futures):
                res_tmp = future.result()
                index = futures[future]
                sub_res[index] = res_tmp
                pbar.update()  # 更新进度条

                # ## debug info
                # total_request_recent, success_request_recent = agent.get_request_num_recent()
                # total_tokens, prompt_tokens, comple_tokens = agent.get_used_tokens()
                # total_request, success_request = agent.get_request_num()
                # pbar.set_postfix_str(f"最近5分钟请求：{total_request_recent}/{success_request_recent}，总请求：{total_request}/{success_request}，总消耗：{total_tokens}，总消耗（输入）：{prompt_tokens}，总消耗（输出）：{comple_tokens}")


        sub_res = [item for sublist in sub_res for item in sublist]
        res.extend(sub_res)

    return res