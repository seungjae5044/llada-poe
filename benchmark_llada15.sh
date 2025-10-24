#!/bin/bash

# Ctrl+C 핸들러 추가
cleanup() {
    echo ""
    echo "========================================="
    echo "Interrupted! Cleaning up..."
    echo "========================================="
    # 모든 eval_llada.py 프로세스 종료
    pkill -f "eval_llada.py"
    exit 130
}

trap cleanup SIGINT SIGTERM

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

BLOCK_LENGTH=32
RESULTS_DIR="benchmark_results"
mkdir -p ${RESULTS_DIR}
mkdir -p ${RESULTS_DIR}/logs

# GPU 스케줄링 설정
NUM_GPUS=16
MAX_PARALLEL_JOBS=16  # 동시 실행할 작업 수 (GPU 개수와 동일)

# 작업 목록 배열
declare -a JOBS
declare -a JOB_NAMES

# 작업 추가 함수
add_job() {
    local job_name="$1"
    local cmd="$2"
    JOBS+=("$cmd")
    JOB_NAMES+=("$job_name")
}

# 동적 병렬 실행 함수 - 작업이 끝나는 즉시 다음 작업 시작
run_dynamic_parallel() {
    echo "========================================="
    echo "Starting dynamic GPU scheduling"
    echo "Total jobs: ${#JOBS[@]}"
    echo "Available GPUs: $NUM_GPUS"
    echo "========================================="

    # GPU 상태 추적 배열 (0: free, PID: busy)
    declare -A gpu_status
    declare -A pid_to_gpu
    declare -A pid_to_job_name

    # 모든 GPU를 free로 초기화
    for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
        gpu_status[$gpu_id]=0
    done

    local job_idx=0
    local completed=0
    local failed=0

    # 작업이 남아있거나 실행 중인 작업이 있는 동안 계속
    while [ $job_idx -lt ${#JOBS[@]} ] || [ ${#pid_to_gpu[@]} -gt 0 ]; do

        # 완료된 작업 확인 및 GPU 해제
        for pid in "${!pid_to_gpu[@]}"; do
            if ! kill -0 $pid 2>/dev/null; then
                # 프로세스가 종료됨
                wait $pid
                local status=$?
                local gpu_id=${pid_to_gpu[$pid]}
                local job_name=${pid_to_job_name[$pid]}

                if [ $status -eq 0 ]; then
                    echo "[GPU $gpu_id] ✓ Completed: $job_name"
                    completed=$((completed+1))
                else
                    echo "[GPU $gpu_id] ✗ Failed (status $status): $job_name"
                    failed=$((failed+1))
                fi

                # GPU 해제
                gpu_status[$gpu_id]=0
                unset pid_to_gpu[$pid]
                unset pid_to_job_name[$pid]
            fi
        done

        # 사용 가능한 GPU 찾기 및 새 작업 시작
        # 동시 실행 작업 수 제한 확인
        local running_jobs=${#pid_to_gpu[@]}
        if [ $job_idx -lt ${#JOBS[@]} ] && [ $running_jobs -lt $MAX_PARALLEL_JOBS ]; then
            for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
                # 동시 실행 제한 재확인
                running_jobs=${#pid_to_gpu[@]}
                if [ $running_jobs -ge $MAX_PARALLEL_JOBS ]; then
                    break
                fi

                if [ ${gpu_status[$gpu_id]} -eq 0 ] && [ $job_idx -lt ${#JOBS[@]} ]; then
                    # GPU가 비어있고 작업이 남아있음
                    echo "[GPU $gpu_id] Starting: ${JOB_NAMES[$job_idx]} (Job $((job_idx+1))/${#JOBS[@]})"

                    CUDA_VISIBLE_DEVICES=$gpu_id bash -c "${JOBS[$job_idx]}" &
                    local pid=$!

                    gpu_status[$gpu_id]=$pid
                    pid_to_gpu[$pid]=$gpu_id
                    pid_to_job_name[$pid]=${JOB_NAMES[$job_idx]}

                    job_idx=$((job_idx+1))
                fi
            done
        fi

        # 짧은 대기 (CPU 사용량 줄이기)
        sleep 2
    done

    echo ""
    echo "========================================="
    echo "All jobs finished!"
    echo "Completed: $completed"
    echo "Failed: $failed"
    echo "========================================="
}

# 모든 작업을 배열에 등록
for GEN_LENGTH in 128 256; do
    STEPS_BASELINE=${GEN_LENGTH}
    STEPS_FAST=$((GEN_LENGTH / BLOCK_LENGTH))

    # GSM8K tasks
    add_job "llada15_gsm8k_${GEN_LENGTH}_baseline" \
        "uv run accelerate launch eval_llada.py --tasks gsm8k --num_fewshot 5 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_BASELINE},block_length=${BLOCK_LENGTH},show_speed=True --output_path ${RESULTS_DIR}/llada15_gsm8k_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_gsm8k_${GEN_LENGTH}_baseline.log"

    add_job "llada15_gsm8k_${GEN_LENGTH}_cache" \
        "uv run accelerate launch eval_llada.py --tasks gsm8k --num_fewshot 5 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True --output_path ${RESULTS_DIR}/llada15_gsm8k_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_gsm8k_${GEN_LENGTH}_cache.log"

    add_job "llada15_gsm8k_Top-8_${GEN_LENGTH}" \
        "uv run accelerate launch eval_llada.py --tasks gsm8k --num_fewshot 5 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,reuse_topk=True,topk_k=8,alpha_base=0.3,show_speed=True --output_path ${RESULTS_DIR}/llada15_gsm8k_Top-8_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_gsm8k_Top-8_${GEN_LENGTH}.log"

    add_job "llada15_gsm8k_Top-8_${GEN_LENGTH}_invert" \
        "uv run accelerate launch eval_llada.py --tasks gsm8k --num_fewshot 5 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,reuse_topk=True,topk_k=8,alpha_base=0.3,show_speed=True,invert=True --output_path ${RESULTS_DIR}/llada15_gsm8k_Top-8_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_gsm8k_Top-8_${GEN_LENGTH}_invert.log"

    add_job "llada15_gsm8k_Top-16_${GEN_LENGTH}" \
        "uv run accelerate launch eval_llada.py --tasks gsm8k --num_fewshot 5 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,reuse_topk=True,topk_k=16,alpha_base=0.3,show_speed=True --output_path ${RESULTS_DIR}/llada15_gsm8k_Top-16_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_gsm8k_Top-16_${GEN_LENGTH}.log"

    add_job "llada15_gsm8k_Top-16_${GEN_LENGTH}_invert" \
        "uv run accelerate launch eval_llada.py --tasks gsm8k --num_fewshot 5 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,reuse_topk=True,topk_k=16,alpha_base=0.3,show_speed=True,invert=True --output_path ${RESULTS_DIR}/llada15_gsm8k_Top-16_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_gsm8k_Top-16_${GEN_LENGTH}_invert.log"

    # HumanEval tasks
    add_job "llada15_humaneval_${GEN_LENGTH}_baseline" \
        "uv run accelerate launch eval_llada.py --tasks humaneval --num_fewshot 0 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_BASELINE},block_length=${BLOCK_LENGTH},show_speed=True --output_path ${RESULTS_DIR}/llada15_humaneval_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_humaneval_${GEN_LENGTH}_baseline.log"

    add_job "llada15_humaneval_${GEN_LENGTH}_cache" \
        "uv run accelerate launch eval_llada.py --tasks humaneval --num_fewshot 0 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True --output_path ${RESULTS_DIR}/llada15_humaneval_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_humaneval_${GEN_LENGTH}_cache.log"

    add_job "llada15_humaneval_Top-8_${GEN_LENGTH}" \
        "uv run accelerate launch eval_llada.py --tasks humaneval --num_fewshot 0 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,reuse_topk=True,topk_k=8,alpha_base=0.3,show_speed=True --output_path ${RESULTS_DIR}/llada15_humaneval_Top-8_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_humaneval_Top-8_${GEN_LENGTH}.log"

    add_job "llada15_humaneval_Top-8_${GEN_LENGTH}_invert" \
        "uv run accelerate launch eval_llada.py --tasks humaneval --num_fewshot 0 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,reuse_topk=True,topk_k=8,alpha_base=0.3,show_speed=True,invert=True --output_path ${RESULTS_DIR}/llada15_humaneval_Top-8_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_humaneval_Top-8_${GEN_LENGTH}_invert.log"

    add_job "llada15_humaneval_Top-16_${GEN_LENGTH}" \
        "uv run accelerate launch eval_llada.py --tasks humaneval --num_fewshot 0 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,reuse_topk=True,topk_k=16,alpha_base=0.3,show_speed=True --output_path ${RESULTS_DIR}/llada15_humaneval_Top-16_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_humaneval_Top-16_${GEN_LENGTH}.log"

    add_job "llada15_humaneval_Top-16_${GEN_LENGTH}_invert" \
        "uv run accelerate launch eval_llada.py --tasks humaneval --num_fewshot 0 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,reuse_topk=True,topk_k=16,alpha_base=0.3,show_speed=True,invert=True --output_path ${RESULTS_DIR}/llada15_humaneval_Top-16_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_humaneval_Top-16_${GEN_LENGTH}_invert.log"

    # MBPP tasks
    add_job "llada15_mbpp_${GEN_LENGTH}_baseline" \
        "uv run accelerate launch eval_llada.py --tasks mbpp --num_fewshot 3 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_BASELINE},block_length=${BLOCK_LENGTH},show_speed=True --output_path ${RESULTS_DIR}/llada15_mbpp_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_mbpp_${GEN_LENGTH}_baseline.log"

    add_job "llada15_mbpp_${GEN_LENGTH}_cache" \
        "uv run accelerate launch eval_llada.py --tasks mbpp --num_fewshot 3 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True --output_path ${RESULTS_DIR}/llada15_mbpp_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_mbpp_${GEN_LENGTH}_cache.log"

    add_job "llada15_mbpp_Top-8_${GEN_LENGTH}" \
        "uv run accelerate launch eval_llada.py --tasks mbpp --num_fewshot 3 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,reuse_topk=True,topk_k=8,alpha_base=0.3,show_speed=True --output_path ${RESULTS_DIR}/llada15_mbpp_Top-8_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_mbpp_Top-8_${GEN_LENGTH}.log"

    add_job "llada15_mbpp_Top-8_${GEN_LENGTH}_invert" \
        "uv run accelerate launch eval_llada.py --tasks mbpp --num_fewshot 3 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,reuse_topk=True,topk_k=8,alpha_base=0.3,show_speed=True,invert=True --output_path ${RESULTS_DIR}/llada15_mbpp_Top-8_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_mbpp_Top-8_${GEN_LENGTH}_invert.log"

    add_job "llada15_mbpp_Top-16_${GEN_LENGTH}" \
        "uv run accelerate launch eval_llada.py --tasks mbpp --num_fewshot 3 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,reuse_topk=True,topk_k=16,alpha_base=0.3,show_speed=True --output_path ${RESULTS_DIR}/llada15_mbpp_Top-16_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_mbpp_Top-16_${GEN_LENGTH}.log"

    add_job "llada15_mbpp_Top-16_${GEN_LENGTH}_invert" \
        "uv run accelerate launch eval_llada.py --tasks mbpp --num_fewshot 3 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,reuse_topk=True,topk_k=16,alpha_base=0.3,show_speed=True,invert=True --output_path ${RESULTS_DIR}/llada15_mbpp_Top-16_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_mbpp_Top-16_${GEN_LENGTH}_invert.log"

    # Minerva Math tasks
    add_job "llada15_minerva_math_${GEN_LENGTH}_baseline" \
        "uv run accelerate launch eval_llada.py --tasks minerva_math --num_fewshot 4 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_BASELINE},block_length=${BLOCK_LENGTH},show_speed=True --output_path ${RESULTS_DIR}/llada15_minerva_math_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_minerva_math_${GEN_LENGTH}_baseline.log"

    add_job "llada15_minerva_math_${GEN_LENGTH}_cache" \
        "uv run accelerate launch eval_llada.py --tasks minerva_math --num_fewshot 4 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True --output_path ${RESULTS_DIR}/llada15_minerva_math_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_minerva_math_${GEN_LENGTH}_cache.log"

    add_job "llada15_minerva_math_Top-8_${GEN_LENGTH}" \
        "uv run accelerate launch eval_llada.py --tasks minerva_math --num_fewshot 4 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,reuse_topk=True,topk_k=8,alpha_base=0.3,show_speed=True --output_path ${RESULTS_DIR}/llada15_minerva_math_Top-8_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_minerva_math_Top-8_${GEN_LENGTH}.log"

    add_job "llada15_minerva_math_Top-8_${GEN_LENGTH}_invert" \
        "uv run accelerate launch eval_llada.py --tasks minerva_math --num_fewshot 4 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,reuse_topk=True,topk_k=8,alpha_base=0.3,show_speed=True,invert=True --output_path ${RESULTS_DIR}/llada15_minerva_math_Top-8_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_minerva_math_Top-8_${GEN_LENGTH}_invert.log"

    add_job "llada15_minerva_math_Top-16_${GEN_LENGTH}" \
        "uv run accelerate launch eval_llada.py --tasks minerva_math --num_fewshot 4 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,reuse_topk=True,topk_k=16,alpha_base=0.3,show_speed=True --output_path ${RESULTS_DIR}/llada15_minerva_math_Top-16_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_minerva_math_Top-16_${GEN_LENGTH}.log"

    add_job "llada15_minerva_math_Top-16_${GEN_LENGTH}_invert" \
        "uv run accelerate launch eval_llada.py --tasks minerva_math --num_fewshot 4 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-1.5',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,reuse_topk=True,topk_k=16,alpha_base=0.3,show_speed=True,invert=True --output_path ${RESULTS_DIR}/llada15_minerva_math_Top-16_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_minerva_math_Top-16_${GEN_LENGTH}_invert.log"
done

# 동적 스케줄링으로 모든 작업 실행
run_dynamic_parallel

echo ""
echo "========================================="
echo "All benchmarks completed!"
echo "Results saved in: ${RESULTS_DIR}"
echo "========================================="