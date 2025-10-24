#!/bin/bash
# 테스트용 벤치마크 스크립트 (A6000 5개)
# 동적 GPU 스케줄링 동작 확인용

# Ctrl+C 핸들러 추가
cleanup() {
    echo ""
    echo "========================================="
    echo "Interrupted! Cleaning up..."
    echo "========================================="
    # 모든 자식 프로세스 종료
    pkill -f "eval_llada.py"
    exit 130
}

trap cleanup SIGINT SIGTERM

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

BLOCK_LENGTH=32
GEN_LENGTH=128  # 테스트용으로 128만 사용
STEPS_BASELINE=${GEN_LENGTH}
STEPS_FAST=$((GEN_LENGTH / BLOCK_LENGTH))

RESULTS_DIR="test_results"
mkdir -p ${RESULTS_DIR}
mkdir -p ${RESULTS_DIR}/logs

# GPU 스케줄링 설정
NUM_GPUS=5  # A6000 5개
MAX_PARALLEL_JOBS=5  # 동시 실행 제한 (메모리 및 다운로드 안정성)

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
    echo "Starting dynamic GPU scheduling TEST"
    echo "Total jobs: ${#JOBS[@]}"
    echo "Available GPUs: $NUM_GPUS"
    echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
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
    local start_time=$(date +%s)

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
                    echo "[$(date +%H:%M:%S)] [GPU $gpu_id] ✓ Completed: $job_name"
                    completed=$((completed+1))
                else
                    echo "[$(date +%H:%M:%S)] [GPU $gpu_id] ✗ Failed (status $status): $job_name"
                    failed=$((failed+1))
                fi

                # GPU 해제
                gpu_status[$gpu_id]=0
                unset pid_to_gpu[$pid]
                unset pid_to_job_name[$pid]

                # 진행 상황 표시
                local total_done=$((completed + failed))
                echo "    Progress: $total_done/${#JOBS[@]} jobs done ($completed succeeded, $failed failed)"
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
                    echo "[$(date +%H:%M:%S)] [GPU $gpu_id] ▶ Starting: ${JOB_NAMES[$job_idx]} (Job $((job_idx+1))/${#JOBS[@]})"

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

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo ""
    echo "========================================="
    echo "TEST COMPLETED!"
    echo "Completed: $completed"
    echo "Failed: $failed"
    echo "Total time: $duration seconds"
    echo "========================================="
}

# 테스트용 작업 등록 (각 task의 baseline과 cache만)
echo "Registering test jobs..."

# GSM8K - 2 jobs
add_job "llada15_gsm8k_${GEN_LENGTH}_baseline" \
    "uv run accelerate launch eval_llada.py --tasks gsm8k --num_fewshot 5 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${GEN_LENGTH},steps=${STEPS_BASELINE},block_length=${BLOCK_LENGTH},show_speed=True --output_path ${RESULTS_DIR}/llada15_gsm8k_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_gsm8k_${GEN_LENGTH}_baseline.log"

add_job "llada15_gsm8k_${GEN_LENGTH}_cache" \
    "uv run accelerate launch eval_llada.py --tasks gsm8k --num_fewshot 5 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True --output_path ${RESULTS_DIR}/llada15_gsm8k_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_gsm8k_${GEN_LENGTH}_cache.log"

# HumanEval - 2 jobs
add_job "llada15_humaneval_${GEN_LENGTH}_baseline" \
    "uv run accelerate launch eval_llada.py --tasks humaneval --num_fewshot 0 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${GEN_LENGTH},steps=${STEPS_BASELINE},block_length=${BLOCK_LENGTH},show_speed=True --output_path ${RESULTS_DIR}/llada15_humaneval_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_humaneval_${GEN_LENGTH}_baseline.log"

add_job "llada15_humaneval_${GEN_LENGTH}_cache" \
    "uv run accelerate launch eval_llada.py --tasks humaneval --num_fewshot 0 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True --output_path ${RESULTS_DIR}/llada15_humaneval_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_humaneval_${GEN_LENGTH}_cache.log"

# MBPP - 2 jobs
add_job "llada15_mbpp_${GEN_LENGTH}_baseline" \
    "uv run accelerate launch eval_llada.py --tasks mbpp --num_fewshot 3 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${GEN_LENGTH},steps=${STEPS_BASELINE},block_length=${BLOCK_LENGTH},show_speed=True --output_path ${RESULTS_DIR}/llada15_mbpp_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_mbpp_${GEN_LENGTH}_baseline.log"

add_job "llada15_mbpp_${GEN_LENGTH}_cache" \
    "uv run accelerate launch eval_llada.py --tasks mbpp --num_fewshot 3 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True --output_path ${RESULTS_DIR}/llada15_mbpp_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_mbpp_${GEN_LENGTH}_cache.log"

# Minerva Math - 2 jobs
add_job "llada15_minerva_math_${GEN_LENGTH}_baseline" \
    "uv run accelerate launch eval_llada.py --tasks minerva_math --num_fewshot 4 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${GEN_LENGTH},steps=${STEPS_BASELINE},block_length=${BLOCK_LENGTH},show_speed=True --output_path ${RESULTS_DIR}/llada15_minerva_math_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_minerva_math_${GEN_LENGTH}_baseline.log"

add_job "llada15_minerva_math_${GEN_LENGTH}_cache" \
    "uv run accelerate launch eval_llada.py --tasks minerva_math --num_fewshot 4 --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${GEN_LENGTH},steps=${STEPS_FAST},block_length=${BLOCK_LENGTH},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True --output_path ${RESULTS_DIR}/llada15_minerva_math_${GEN_LENGTH} --log_samples 2>&1 | tee ${RESULTS_DIR}/logs/llada15_minerva_math_${GEN_LENGTH}_cache.log"

echo "Total jobs registered: ${#JOBS[@]}"
echo ""

# 동적 스케줄링으로 모든 작업 실행
run_dynamic_parallel

echo ""
echo "========================================="
echo "Test benchmark completed!"
echo "Results saved in: ${RESULTS_DIR}"
echo "========================================="
