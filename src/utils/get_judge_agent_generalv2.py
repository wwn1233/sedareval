import os
import yaml
import tiktoken
from .utils import logger
from .api_general_v2 import OpenAIGPT_General_V2

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

############################################### GPT Agent
def get_agent_generalv2(api_key=None, force_gpt4_turbo=False):
    if os.getenv('CONFIG_JUDGE') is None:
        config_file = os.path.join(CONFIG_BASE, "configs", "config_openai.yaml")
    else:
        config_path = os.getenv('CONFIG_JUDGE')
        if os.path.exists(config_path):
            config_file = config_path
            print("环境变量'CONFIG_JUDGE'已设置，其值为：", config_path)
        else:
            config_file = os.path.join(CONFIG_BASE, "configs", "config_openai.yaml")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        logger.info(f"Start GPT agent with config:\n{config}\n{'-'*100}")
        batch_size = config.pop('batch', 20)
        system_message = config.pop('sys_prompt', "You're a helpful assistant.")  ##默认统一个system_prompt
        query_message = config.pop('prompt', "")
        query_field = config.pop('query_field', 'query')
        gpt_config = config.pop('gpt')
        temperature = gpt_config.pop('temperature')
        max_tokens = gpt_config.pop('max_tokens')
        vote_times = gpt_config.pop('vote_times')
        top_p = gpt_config.pop('top_p')
        # best_of = gpt_config.pop('best_of')
        mode = config.pop('mode',0)

    agent = OpenAIGPT_General_V2(temperature=temperature, max_tokens=max_tokens, vote_times=vote_times, top_p=top_p, batch_size=batch_size, mode=mode, **gpt_config)

    return agent


get_all_agent_generalv2 = {
    "创作": get_agent_generalv2,
    "多轮": get_agent_generalv2,
    "工具使用": get_agent_generalv2,
    "单轮客观": get_agent_generalv2
}

def get_all_agents_generalv2():
    return get_all_agent_generalv2



