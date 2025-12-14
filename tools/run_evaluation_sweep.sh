#!/bin/bash

# Evaluation sweep script for JunRAG pipelines
# Tests multiple pipelines with different parameter combinations
#
# Usage:
#   ./tools/run_evaluation_sweep.sh
#
# Environment variables (optional):
#   METADATA_PATH - Path to Qdrant metadata JSON file (default: data/metadata/test_late_metadata.json)
#   TENSOR_PARALLEL_SIZE - Number of GPUs (default: 8)
#   GPU_MEMORY_UTILIZATION - GPU memory fraction (default: 0.9)
#
# To customize:
#   1. Edit the arrays below to add more models or parameter values
#   2. Modify PIPELINES array to test specific pipelines only
#   3. Adjust parameter arrays (RETRIEVAL_TOP_K_VALUES, etc.) as needed

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVALUATE_SCRIPT="$SCRIPT_DIR/evaluate.py"

# Check if evaluate.py exists
if [ ! -f "$EVALUATE_SCRIPT" ]; then
    echo -e "${RED}Error: evaluate.py not found at $EVALUATE_SCRIPT${NC}" >&2
    exit 1
fi

# Default values (can be overridden via command line or environment)
METADATA_PATH="${METADATA_PATH:-data/metadata/test_late_metadata.json}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"

# Model arrays (defaults only for now - can be extended later)
EMBEDDING_MODELS=("jinaai/jina-embeddings-v3")
RERANKER_MODELS=("Qwen/Qwen3-Reranker-4B")
LLM_MODELS=("gpt-5.1-2025-11-13")
# LLM_MODELS=("gpt-5.2-2025-12-11")
DECOMPOSITION_MODELS=("gpt-4o")

# Parameter arrays to test
# Note: rerank_top_k and chunks_per_subquery must be <= retrieval_top_k
# Invalid combinations will be automatically skipped during execution
RETRIEVAL_TOP_K_VALUES=(10 20 30)
RERANK_TOP_K_VALUES=(5 10 20)  # For naive pipeline (must be <= retrieval_top_k)
CHUNKS_PER_SUBQUERY_VALUES=(5 10 20)  # For parallel/sequential_decomposition (must be <= retrieval_top_k)
RERANK_BATCH_SIZE_VALUES=(32 64 128)

# Pipelines to test
# PIPELINES=("naive" "parallel" "sequential_decomposition")
PIPELINES=("sequential_decomposition")

# Log file for tracking runs
LOG_FILE="${PROJECT_ROOT}/results/evaluation_sweep_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Function to run evaluation
run_evaluation() {
    local pipeline=$1
    local embedding_model=$2
    local reranker_model=$3
    local llm_model=$4
    local decomposition_model=$5
    local retrieval_top_k=$6
    local rerank_top_k=$7
    local chunks_per_subquery=$8
    local rerank_batch_size=$9

    log "=========================================="
    log "Starting evaluation:"
    log "  Pipeline: $pipeline"
    log "  Embedding Model: $embedding_model"
    log "  Reranker Model: $reranker_model"
    log "  LLM Model: $llm_model"
    log "  Decomposition Model: $decomposition_model"
    log "  Retrieval Top-K: $retrieval_top_k"
    log "  Rerank Top-K: $rerank_top_k"
    log "  Chunks Per Subquery: $chunks_per_subquery"
    log "  Rerank Batch Size: $rerank_batch_size"
    log "=========================================="

    # Build command arguments
    local cmd_args=(
        --pipeline "$pipeline"
        --metadata_path "$METADATA_PATH"
        --embedding_model "$embedding_model"
        --reranker_model "$reranker_model"
        --llm_model "$llm_model"
        --decomposition_model "$decomposition_model"
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
        --max_model_len "$MAX_MODEL_LEN"
        --retrieval_top_k "$retrieval_top_k"
        --rerank_batch_size "$rerank_batch_size"
        --temperature "$TEMPERATURE"
        --max_tokens "$MAX_TOKENS"
        --reasoning_effort "$REASONING_EFFORT"
    )

    # Add pipeline-specific parameters
    if [ "$pipeline" == "naive" ]; then
        cmd_args+=(--rerank_top_k "$rerank_top_k")
    else
        cmd_args+=(--chunks_per_subquery "$chunks_per_subquery")
    fi

    # Run the evaluation
    cd "$PROJECT_ROOT"
    if python3 "$EVALUATE_SCRIPT" "${cmd_args[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        echo -e "${GREEN}✓ Evaluation completed successfully${NC}" | tee -a "$LOG_FILE"
        return 0
    else
        echo -e "${RED}✗ Evaluation failed${NC}" | tee -a "$LOG_FILE"
        return 1
    fi
}

# Main execution
main() {
    log "Starting evaluation sweep"
    log "Log file: $LOG_FILE"
    log "Project root: $PROJECT_ROOT"
    log ""

    # Calculate and display total number of runs
    # Note: This is an estimate - actual runs may be fewer due to validation
    local naive_runs=$((${#RETRIEVAL_TOP_K_VALUES[@]} * ${#RERANK_TOP_K_VALUES[@]} * ${#RERANK_BATCH_SIZE_VALUES[@]}))
    local other_runs=$((${#RETRIEVAL_TOP_K_VALUES[@]} * ${#CHUNKS_PER_SUBQUERY_VALUES[@]} * ${#RERANK_BATCH_SIZE_VALUES[@]}))
    local total_combinations=$((${#EMBEDDING_MODELS[@]} * ${#RERANKER_MODELS[@]} * ${#LLM_MODELS[@]} * ${#DECOMPOSITION_MODELS[@]}))
    local naive_total=$((naive_runs * total_combinations))
    local parallel_total=$((other_runs * total_combinations))
    local sequential_total=$((other_runs * total_combinations))
    local grand_total=$((naive_total + parallel_total + sequential_total))

    log "Note: Invalid combinations (e.g., rerank_top_k > retrieval_top_k) will be skipped"

    log "Configuration summary:"
    log "  Pipelines: ${PIPELINES[*]}"
    log "  Embedding models: ${EMBEDDING_MODELS[*]}"
    log "  Reranker models: ${RERANKER_MODELS[*]}"
    log "  LLM models: ${LLM_MODELS[*]}"
    log "  Decomposition models: ${DECOMPOSITION_MODELS[*]}"
    log "  Retrieval Top-K values: ${RETRIEVAL_TOP_K_VALUES[*]}"
    log "  Rerank Top-K values (naive): ${RERANK_TOP_K_VALUES[*]}"
    log "  Chunks per subquery (parallel/sequential): ${CHUNKS_PER_SUBQUERY_VALUES[*]}"
    log "  Rerank batch sizes: ${RERANK_BATCH_SIZE_VALUES[*]}"
    log ""
    log "Estimated total runs for each pipeline: $grand_total"
    log ""

    local total_runs=0
    local successful_runs=0
    local failed_runs=0

    # Loop through all combinations
    for pipeline in "${PIPELINES[@]}"; do
        for embedding_model in "${EMBEDDING_MODELS[@]}"; do
            for reranker_model in "${RERANKER_MODELS[@]}"; do
                for llm_model in "${LLM_MODELS[@]}"; do
                    for decomposition_model in "${DECOMPOSITION_MODELS[@]}"; do
                        for retrieval_top_k in "${RETRIEVAL_TOP_K_VALUES[@]}"; do
                            for rerank_batch_size in "${RERANK_BATCH_SIZE_VALUES[@]}"; do
                                if [ "$pipeline" == "naive" ]; then
                                    # For naive pipeline, use rerank_top_k
                                    # Ensure rerank_top_k <= retrieval_top_k
                                    for rerank_top_k in "${RERANK_TOP_K_VALUES[@]}"; do
                                        if [ "$rerank_top_k" -gt "$retrieval_top_k" ]; then
                                            log "Skipping invalid combination: rerank_top_k=$rerank_top_k > retrieval_top_k=$retrieval_top_k"
                                            continue
                                        fi
                                        total_runs=$((total_runs + 1))
                                        if run_evaluation \
                                            "$pipeline" \
                                            "$embedding_model" \
                                            "$reranker_model" \
                                            "$llm_model" \
                                            "$decomposition_model" \
                                            "$retrieval_top_k" \
                                            "$rerank_top_k" \
                                            "N/A" \
                                            "$rerank_batch_size"; then
                                            successful_runs=$((successful_runs + 1))
                                        else
                                            failed_runs=$((failed_runs + 1))
                                        fi
                                        echo ""  # Blank line between runs
                                    done
                                else
                                    # For parallel/sequential_decomposition, use chunks_per_subquery
                                    # Ensure chunks_per_subquery <= retrieval_top_k
                                    for chunks_per_subquery in "${CHUNKS_PER_SUBQUERY_VALUES[@]}"; do
                                        if [ "$chunks_per_subquery" -gt "$retrieval_top_k" ]; then
                                            log "Skipping invalid combination: chunks_per_subquery=$chunks_per_subquery > retrieval_top_k=$retrieval_top_k"
                                            continue
                                        fi
                                        total_runs=$((total_runs + 1))
                                        if run_evaluation \
                                            "$pipeline" \
                                            "$embedding_model" \
                                            "$reranker_model" \
                                            "$llm_model" \
                                            "$decomposition_model" \
                                            "$retrieval_top_k" \
                                            "N/A" \
                                            "$chunks_per_subquery" \
                                            "$rerank_batch_size"; then
                                            successful_runs=$((successful_runs + 1))
                                        else
                                            failed_runs=$((failed_runs + 1))
                                        fi
                                        echo ""  # Blank line between runs
                                    done
                                fi
                            done
                        done
                    done
                done
            done
        done
    done

    # Summary
    log ""
    log "=========================================="
    log "Evaluation sweep completed"
    log "  Total runs: $total_runs"
    log "  Successful: $successful_runs"
    log "  Failed: $failed_runs"
    log "  Log file: $LOG_FILE"
    log "=========================================="
}

# Run main function
main "$@"
