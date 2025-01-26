## Enviroment
if [ -z "$LLM_AS_JUDGE_PATH" ]; then
  export BRANCH=main
  git clone -b ${BRANCH} https://github.com/wwn1233/sedareval.git
  export LLM_AS_JUDGE_PATH="$(pwd)/sedareval"
else
  echo $LLM_AS_JUDGE_PATH
fi

cd $LLM_AS_JUDGE_PATH


## custom parameters.
########################################### you can set these variables ###########################################
# 环境变量： MODEL_NAME   MODEL_DIR  SYSTEM_PROMPT  INPUT_FILE  OUTPUT_PREFIX    JUDGE_OUTPUT_PREFIX   LOG_DIR
if [ -z "$EVAL_JSONLS" ] || [ -z "$OUTPUT_PREFIX_JUDGE" ]; then
    echo "One or more required environment variables are not set."
    exit 1
fi

eval_jsons=$EVAL_JSONLS
output_prefix=$OUTPUT_PREFIX_JUDGE

echo "EVAL_JSONLS is set to: $eval_jsons"
echo "OUTPUT_PREFIX_JUDGE is set to: $output_prefix"

DATASEET_TEST_SET=$LLM_AS_JUDGE_PATH/dataset/sedareval_data_ch.jsonl
DATASET_TEST_AUTOPROMPT=$LLM_AS_JUDGE_PATH/dataset/sedareval_data_ch.jsonl
EVAL_DOMENSIONS=None
EVAL_JSONLS=$eval_jsons
OUTPUT_PREFIX_JUDGE=$output_prefix
EVAL_ID=None
API_KEY=None
API_TYPE="generalv2"

cd $LLM_AS_JUDGE_PATH
echo "Processing"$EVAL_JSONLS
python ./src/main.py --dataset_test_set $DATASEET_TEST_SET --dataset_test_aotoprompt $DATASET_TEST_AUTOPROMPT  --eval_dimensions $EVAL_DOMENSIONS --eval_ids $EVAL_ID --eval_jsonls $EVAL_JSONLS, --outout_prefix $OUTPUT_PREFIX_JUDGE --api_key $API_KEY --api_type $API_TYPE --develop_mode --force_recreate 