git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip3 install deepspeed
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LLAMA_FACTORY_PATH=$(pwd)
cd ..

export BRANCH=main
git clone -b ${BRANCH} https://github.com/wwn1233/sedareval.git
export LLM_AS_JUDGE_PATH="$(pwd)/sedareval"
cd $LLM_AS_JUDGE_PATH


export MODEL_NAME=Qwen2.5-7B-Instruct
export MODEL_DIR=< the path to Qwen2.5-7B-Instruct >
export SYSTEM_PROMPT="You are a helpful and harmless assistant. You are Qwen developed by Alibaba."
export INPUT_FILE=dataset/sedareval_data_ch.jsonl
export OUTPUT_PREFIX=results/test
export JUDGE_OUTPUT_PREFIX=results/test/$MODEL_NAME
export LOG_DIR=results/test/$MODEL_NAME/logs

bash ./scripts/run_infer_eval_gpt4o-as-judge.sh