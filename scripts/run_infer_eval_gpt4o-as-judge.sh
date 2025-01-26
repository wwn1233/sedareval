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
if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_DIR" ] || [ -z "$SYSTEM_PROMPT" ] || [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_PREFIX" ] || [ -z "$JUDGE_OUTPUT_PREFIX" ]; then
    echo "One or more required environment variables are not set."
    exit 1
fi

model_name=$MODEL_NAME
model_dir=$MODEL_DIR
system_prompt=$SYSTEM_PROMPT
input_file=$INPUT_FILE
output_prefix=$OUTPUT_PREFIX
judge_output_prefix=$JUDGE_OUTPUT_PREFIX
log_dir=$LOG_DIR


echo "MODEL_NAME is set to: $model_name"
echo "MODEL_DIR is set to: $model_dir"
echo "SYSTEM_PROMPT is set to: $system_prompt"
echo "INPUT_FILE is set to: $input_file"
echo "OUTPUT_PREFIX is set to: $output_prefix"
echo "JUDGE_OUTPUT_PREFIX is set to: $judge_output_prefix"
echo "LOG_DIR is set to: $log_dir"

# echo "$(pwd)"
mkdir -p $log_dir
if [[ "$log_dir" == /* ]]; then
  echo "log_dir is: $log_dir"
else
  log_dir=$LLM_AS_JUDGE_PATH/$log_dir
  echo "log_dir is set to: $log_dir"
fi
if [[ "$input_file" == /* ]]; then
  echo "input_file is: $input_file"
else
  input_file=$LLM_AS_JUDGE_PATH/$input_file
  echo "input_file is set to: $input_file"
fi
if [[ "$output_prefix" == /* ]]; then
  echo "output_prefix is: $output_prefix"
else
  output_prefix=$LLM_AS_JUDGE_PATH/$output_prefix
  echo "output_prefix is set to: $output_prefix"
fi

gen_parallel=8
deploy_yaml=$LLM_AS_JUDGE_PATH/src/configs/vllm/deploy_vllm.yaml
api_yaml=$LLM_AS_JUDGE_PATH/src/configs/vllm/api_config.yaml
gen_answer_yaml=$LLM_AS_JUDGE_PATH/src/configs/vllm/gen_answer_config.yaml

function start_vllm() {
    cd $LLAMA_FACTORY_PATH
    sed -i.bak "s|^\(model_name_or_path:\).*|\1 $model_dir|" "$deploy_yaml"
    nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 API_PORT=8000 llamafactory-cli api $deploy_yaml" > $log_dir"/1.log" 2>&1 &

    sleep 120
    max_attempts=10
    attempt=0 
    while [ "$attempt" -lt "$max_attempts" ]; do
        process_count=$(ps aux | grep '[l]lamafactory-cli api' | wc -l)

        if [ "$process_count" -gt 0 ]; then
            echo "Detected $process_count instance(s) of llamafactory-cli api running."
            break
        else
            echo "Attempt $((attempt + 1)): No llamafactory-cli api processes detected. Checking again in 5 seconds..."
        fi

        attempt=$((attempt + 1))

        sleep 5
    done
    if [ "$attempt" -eq "$max_attempts" ]; then
        echo "Reached maximum attempts without detecting the process."
        exit 1
    fi
}

function infer() {
  cd $LLM_AS_JUDGE_PATH
  target_api_yaml="/tmp/api_"$model_name".yaml"
  cp $api_yaml $target_api_yaml
  target_gen_answer_yaml="/tmp/gen_answer_"$model_name".yaml"
  cp $gen_answer_yaml $target_gen_answer_yaml
  sed -i "s/TEMPLATE/${model_name}/g" "$target_api_yaml"
  sed -i "s/TEMPLATE/${model_name}/g" "$target_gen_answer_yaml"
  
  sed -i "s/system_prompt: .*/system_prompt: ${system_prompt}/" "$target_gen_answer_yaml"

  # 生成答案
  python src/infer.py --input-file $input_file --setting-file $target_gen_answer_yaml  --endpoint-file $target_api_yaml --output_prefix $output_prefix --parallel $gen_parallel

#   pids=$(ps aux | grep '[l]lamafactory-cli api' | awk '{print $2}')
#   if [ -z "$pids" ]; then
#     echo "No llamafactory-cli api processes found."
#   else
#     echo "Killing llamafactory-cli api processes: $pids"
#     kill $pids
#     sleep 10
#     echo "Processes terminated."
#   fi
#   fuser -v /dev/nvidia*|awk -F " " '{print $0}' >/tmp/pid.file
#   while read pid ; do kill -9 $pid; done </tmp/pid.file

}

function judge() {
    base_name=$(basename "$input_file" .jsonl)
    infer_output_file=$output_prefix/$model_name/$base_name/infer_results.jsonl

    DATASEET_TEST_SET=$LLM_AS_JUDGE_PATH/dataset/sedareval_data_ch.jsonl
    DATASET_TEST_AUTOPROMPT=$LLM_AS_JUDGE_PATH/dataset/sedareval_data_ch.jsonl
    EVAL_DOMENSIONS=None
    EVAL_JSONLS=$infer_output_file
    OUTPUT_PREFIX_JUDGE=$judge_output_prefix
    EVAL_ID=None
    API_KEY=None
    API_TYPE="generalv2"

    cd $LLM_AS_JUDGE_PATH
    echo "Processing"$EVAL_JSONLS
    python ./src/main.py --dataset_test_set $DATASEET_TEST_SET --dataset_test_aotoprompt $DATASET_TEST_AUTOPROMPT  --eval_dimensions $EVAL_DOMENSIONS --eval_ids $EVAL_ID --eval_jsonls $EVAL_JSONLS, --outout_prefix $OUTPUT_PREFIX_JUDGE --api_key $API_KEY --api_type $API_TYPE --develop_mode --force_recreate 
}


## infer eval show
start_vllm
infer
judge


