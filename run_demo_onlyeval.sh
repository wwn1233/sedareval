## Enviroment
export BRANCH=main
git clone -b ${BRANCH} https://github.com/wwn1233/sedareval.git
export LLM_AS_JUDGE_PATH="$(pwd)/sedareval"
cd $LLM_AS_JUDGE_PATH

export EVAL_JSONLS=dataset/test.jsonl
export OUTPUT_PREFIX_JUDGE=results/test

bash ./scripts/run_eval_gpt4o-as-judge.sh