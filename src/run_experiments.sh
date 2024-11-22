#!/bin/bash

# Exit on any error
set -e

# Exit on any undefined variable
set -u

# Exit if any command in a pipe fails
set -o pipefail

# Change to project root directory
cd "$(dirname "$0")/.."

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Memory management settings
MAX_MEMORY_PERCENT=80
CLEANUP_INTERVAL=300  # 5 minutes
BATCH_PROCESSING_SIZE=1000

# Function to log messages with timestamps and colors
log_message() {
    local level=$1
    local message=$2
    local timestamp
    timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")
            printf "${GREEN}[%s] [INFO] %s${NC}\n" "$timestamp" "$message"
            ;;
        "WARNING")
            printf "${YELLOW}[%s] [WARNING] %s${NC}\n" "$timestamp" "$message"
            ;;
        "ERROR")
            printf "${RED}[%s] [ERROR] %s${NC}\n" "$timestamp" "$message"
            ;;
        *)
            printf "[%s] %s\n" "$timestamp" "$message"
            ;;
    esac
}

# Function to check available memory
check_memory() {
    local memory_used
    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS
        memory_used=$(ps -A -o %mem | awk '{mem += $1} END {print mem}')
    else
        # Linux
        memory_used=$(free | grep Mem | awk '{print $3/$2 * 100}')
    fi
    
    echo "${memory_used%.*}"
}

# Function to cleanup memory
cleanup_memory() {
    log_message "INFO" "Performing memory cleanup..."
    
    # Clear system caches if running as root
    if [ "$(id -u)" = "0" ]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches
    fi
    
    # Clear Python cache directories
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    
    # Remove temporary files
    find /tmp -name "experiment_*" -mmin +60 -delete 2>/dev/null || true
    
    # Clear GPU memory if NVIDIA GPU is present
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --gpu-reset 2>/dev/null || true
    fi
    
    log_message "INFO" "Memory cleanup completed"
}

# Function to monitor memory usage and trigger cleanup
monitor_memory() {
    local memory_used
    memory_used=$(check_memory)
    
    if [ "${memory_used%.*}" -gt "$MAX_MEMORY_PERCENT" ]; then
        log_message "WARNING" "High memory usage detected (${memory_used}%). Triggering cleanup..."
        cleanup_memory
    fi
}


# Function to initialize indices for a retriever with batch processing
initialize_indices() {
    local retriever=$1
    local experiment=$2
    
    log_message "INFO" "Initializing indices for $retriever retriever in $experiment experiment..."
    
    # Create temporary directory for indices
    local temp_dir="data/mappings/temp/${experiment}/${retriever}"
    mkdir -p "$temp_dir"
    
    # Process in batches
    local start_idx=0
    while true; do
        log_message "INFO" "Processing batch starting at index $start_idx"
        
        if ! python src/utils/initialize_indices.py \
            --experiment "$experiment" \
            --retriever "$retriever" \
            --batch_size "$BATCH_PROCESSING_SIZE" \
            --start_idx "$start_idx" \
            --output_dir "$temp_dir" \
            --enable_gc true; then
            log_message "ERROR" "Failed to initialize indices for $retriever"
            return 1
        fi
        
        # Check if batch was empty (end of data)
        if [ ! -f "${temp_dir}/batch_${start_idx}.json" ]; then
            break
        fi
        
        # Merge batch results
        if [ -f "${temp_dir}/indices_merged.json" ]; then
            # Merge with existing results
            jq -s '.[0] * .[1]' \
                "${temp_dir}/indices_merged.json" \
                "${temp_dir}/batch_${start_idx}.json" \
                > "${temp_dir}/indices_merged_temp.json"
            mv "${temp_dir}/indices_merged_temp.json" "${temp_dir}/indices_merged.json"
        else
            # First batch
            mv "${temp_dir}/batch_${start_idx}.json" "${temp_dir}/indices_merged.json"
        fi
        
        # Cleanup batch file
        rm -f "${temp_dir}/batch_${start_idx}.json"
        
        # Monitor memory
        monitor_memory
        
        # Move to next batch
        start_idx=$((start_idx + BATCH_PROCESSING_SIZE))
    done
    
    # Finalize indices
    mv "${temp_dir}/indices_merged.json" "${temp_dir}/final_indices.json"
    
    log_message "INFO" "Index initialization completed for $retriever"
    return 0
}

# Function to check index mapping files with batched validation
check_index_mappings() {
    local experiment=$1
    local retriever=$2
    
    log_message "INFO" "Checking index mapping files for ${experiment}/${retriever}..."
    
    local mapping_dir="data/mappings/temp/${experiment}/${retriever}"
    local required_mappings=(
        "${mapping_dir}/corpus_idx_mapping.json"
        "${mapping_dir}/dataset_idx_mapping.json"
        "${mapping_dir}/search_idx_mapping.json"
        "${mapping_dir}/random_idx_mapping.json"
    )
    
    for mapping in "${required_mappings[@]}"; do
        if [ ! -f "$mapping" ]; then
            log_message "WARNING" "Missing index mapping: $mapping"
            return 1
        fi
        
        # Validate file structure in chunks
        if ! python -c "
import json
with open('$mapping') as f:
    chunk_size = 1024 * 1024  # 1MB chunks
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break
        # Try parsing each chunk
        try:
            json.loads(chunk)
        except ValueError as e:
            if 'Extra data' not in str(e):  # Ignore partial chunks
                raise
" 2>/dev/null; then
            log_message "WARNING" "Invalid JSON structure in: $mapping"
            return 1
        fi
    done
    
    log_message "INFO" "Index mapping files verified"
    return 0
}

# Function to check required python packages with version verification
check_python_packages() {
    log_message "INFO" "Checking Python packages..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "power_of_noise_env" ]; then
        log_message "INFO" "Creating virtual environment..."
        python -m venv power_of_noise_env
        source power_of_noise_env/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
    else
        source power_of_noise_env/bin/activate
    fi
    
    required_packages=(
        "torch"
        "transformers" 
        "datasets"
        "rich"
        "numpy"
        "pandas"
        #"scikit-learn"
        #"faiss"
    )
    
    for package in "${required_packages[@]}"; do
        if ! python -c "import $package" &> /dev/null; then
            log_message "ERROR" "Required package not found: $package"
            log_message "INFO" "Please run: pip install -r requirements.txt"
            return 1
        fi
        
        # Check package version
        version=$(pip show "$package" | grep Version | awk '{print $2}')
        log_message "INFO" "$package version: $version"
    done
    
    log_message "INFO" "All required Python packages are installed"
    return 0
}

# Function to check data directories and files with size verification
check_data_directories() {
    log_message "INFO" "Checking data directories..."
    
    required_dirs=(
        "data/datasets"
        "data/corpus"
        "data/corpus/train"
        "data/embeddings"
        "data/mappings"
        "data/random_results"
        "data/retrieval_results"
        "logs"
        "results"
        "logs/indexing"
        "data/mappings/temp"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            log_message "WARNING" "Creating missing directory: $dir"
            mkdir -p "$dir"
        fi
        
        # Check directory size
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        log_message "INFO" "Directory $dir size: $size"
    done

    # Check for required dataset files
    if [ ! -d "data/corpus/train" ] || [ ! -f "data/corpus/dataset_dict.json" ]; then
        log_message "ERROR" "Missing corpus dataset files. Please ensure data/corpus/train contains arrow files"
        return 1
    fi
    
    if [ ! -f "data/datasets/10k_train_dataset.json" ] || [ ! -f "data/datasets/test_dataset.json" ]; then
        log_message "ERROR" "Missing dataset files. Please download the required datasets"
        return 1
    fi
    
    log_message "INFO" "Data directories check completed"
    return 0
}

# Function to check and fix .env file with validation
check_env_file() {
    log_message "INFO" "Checking .env file..."
    if [ ! -f .env ]; then
        log_message "ERROR" "Error: .env file not found"
        return 1
    fi
    
    # Validate required variables
    required_vars=(
        "HF_TOKEN"
        "PYTHONPATH"
        "CUDA_VISIBLE_DEVICES"
    )
    
    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" .env; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        log_message "ERROR" "Missing required variables in .env: ${missing_vars[*]}"
        return 1
    fi
}

# Function to validate environment variables
validate_env_vars() {
    log_message "INFO" "Validating environment variables..."
    
    # Source the .env file
    # shellcheck source=/dev/null
    source .env || {
        log_message "ERROR" "Error: Failed to source .env file"
        return 1
    }
    
    # Validate CUDA visibility if GPU is required
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        if ! command -v nvidia-smi &> /dev/null; then
            log_message "WARNING" "CUDA_VISIBLE_DEVICES is set but no NVIDIA GPU found"
        fi
    fi
}

# Function to run a single experiment with memory monitoring
run_experiment() {
    local exp_name=$1
    local exp_mode=$2
    local retriever_type=$3
    local script_path=$4
    
    log_message "INFO" "Starting $exp_name experiment with $retriever_type retriever..."
    
    if [ ! -f "$script_path" ]; then
        log_message "ERROR" "Error: Script not found: $script_path"
        return 1
    fi
    
    # Initialize indices for this experiment
    if ! initialize_indices "$retriever_type" "$exp_mode"; then
        log_message "ERROR" "Index initialization failed for $exp_name"
        return 1
    fi
    
    # Verify indices
    if ! check_index_mappings "$exp_mode" "$retriever_type"; then
        log_message "ERROR" "Index mapping verification failed for $exp_name"
        return 1
    fi
    
    local start_time
    start_time=$(date +%s)
    
    # Setup memory monitoring in background
    while true; do
        monitor_memory
        sleep "$CLEANUP_INTERVAL"
    done &
    local monitor_pid=$!
    
    # Run experiment
    if ! python "$script_path" \
        --experiment_mode "$exp_mode" \
        --retriever_type "$retriever_type" \
        --batch_size "$BATCH_PROCESSING_SIZE" \
        --index_dir "data/mappings/temp/${exp_mode}/${retriever_type}"; then
        log_message "ERROR" "Error: $exp_name experiment failed"
        kill "$monitor_pid"
        return 1
    fi
    
    # Stop memory monitoring
    kill "$monitor_pid"
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Read and analyze results
    local results_dir="results/${exp_mode}/${retriever_type}"
    python src/data_handlers/read_generation_results.py \
        --results_dir "$results_dir" \
        --retriever_type "$retriever_type" \
        --experiment_mode "$exp_mode" \
        --batch_size "$BATCH_PROCESSING_SIZE" \
        --index_dir "data/mappings/temp/${exp_mode}/${retriever_type}"
    
    log_message "INFO" "✓ $exp_name completed in $duration seconds"
    echo "----------------------------------------"
    
    return 0
}

# Function to cleanup indices with archiving
cleanup_indices() {
    local exp_mode=$1
    local retriever_type=$2
    
    log_message "INFO" "Cleaning up indices for ${exp_mode}/${retriever_type}..."
    
    # Create archive directory with timestamp
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local archive_dir="data/mappings/archive/${exp_mode}/${retriever_type}/${timestamp}"
    mkdir -p "$archive_dir"
    
    # Archive index files in chunks
    find "data/mappings/temp/${exp_mode}/${retriever_type}" -name "*.json" -print0 | \
    while IFS= read -r -d '' file; do
        # Compress file
        gzip -c "$file" > "${archive_dir}/$(basename "$file").gz"
        rm "$file"
        
        # Monitor memory
        monitor_memory
    done
    
    log_message "INFO" "Index cleanup completed"
}

# Function to run all experiments for a retriever
run_retriever_experiments() {
    local retriever_type=$1
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Run each experiment
    run_experiment "Baseline RAG" "baseline" "$retriever_type" "src/experiments/baseline_rag.py"
    cleanup_indices "baseline" "$retriever_type"
    
    run_experiment "KMeans RAG" "kmeans" "$retriever_type" "src/experiments/kmeans_rag.py"
    cleanup_indices "kmeans" "$retriever_type"
    
    run_experiment "Fusion KMeans RAG" "fusion_kmeans" "$retriever_type" "src/experiments/fusion_kmeans.py"

    run_experiment "Fusion Categories RAG" "fusion_categories" "$retriever_type" "src/experiments/fusion_categories.py"
    cleanup_indices "fusion_categories" "$retriever_type"
}

# Function to clean up temporary files with memory monitoring
cleanup() {
    log_message "INFO" "Cleaning up..."
    
    # Remove Python bytecode
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
    
    # Remove temporary experiment files
    find . -type f -name "temp_*" -delete
    find . -type f -name "*.tmp" -delete
    
    # Clean up any remaining memory dumps
    find . -type f -name "core.*" -delete
    
    # Clean GPU memory if available
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --gpu-reset 2>/dev/null || true
    fi
}

# Main execution with error handling and memory monitoring
main() {
    # Check if arguments are provided
    if [ "$#" -lt 1 ]; then
        log_message "ERROR" "Not enough arguments"
        echo "Usage: $0 [--all|--baseline|--kmeans|--fusion] [contriever|bm25|adore|all]"
        exit 1
    fi

    log_message "INFO" "Starting RAG experiments suite with memory optimization"

    # Parse arguments
    local experiment_type="$1"
    local retriever_type="${2:-all}"
    
    # Export batch size for experiments
    export BATCH_PROCESSING_SIZE=1000  # Reduced from original for better memory management
    export MAX_MEMORY_PERCENT=80  # Maximum memory usage before cleanup
    export CLEANUP_INTERVAL=300   # Cleanup interval in seconds

    # Initial checks and setup
    check_env_file
    validate_env_vars
    check_python_packages
    check_data_directories

    # Activate virtual environment
    if [ -d "power_of_noise_env" ]; then
        # shellcheck source=/dev/null
        source power_of_noise_env/bin/activate || {
            log_message "ERROR" "Error: Failed to activate virtual environment"
            exit 1
        }
        log_message "INFO" "Virtual environment activated"
    else
        log_message "WARNING" "Virtual environment not found, proceeding with system Python"
    fi

    # Setup memory monitoring in background
    while true; do
        monitor_memory
        sleep "$CLEANUP_INTERVAL"
    done &
    local monitor_pid=$!

    # Run experiments based on arguments
    case "$experiment_type" in
        --all)
            log_message "INFO" "Running all experiments with memory optimization"
            if [ "$retriever_type" = "all" ]; then
                for r_type in "contriever" "bm25" "adore"; do
                    run_retriever_experiments "$r_type"
                    # Force cleanup between retrievers
                    cleanup_memory
                done
            else
                run_retriever_experiments "$retriever_type"
            fi
            
            # Generate visualizations with memory optimization
            log_message "INFO" "Generating visualizations..."
            python src/visualization/metrics_plotter.py --batch_size "$BATCH_PROCESSING_SIZE"
            ;;
            
        --baseline)
            run_experiment "Baseline RAG" "baseline" "$retriever_type" \
                "src/experiments/baseline_rag.py"
            cleanup_indices "baseline" "$retriever_type"
            ;;
            
        --kmeans)
            run_experiment "KMeans RAG" "kmeans" "$retriever_type" \
                "src/experiments/kmeans_rag.py"
            cleanup_indices "kmeans" "$retriever_type"
            ;;
            
        --fusion)
            log_message "INFO" "Running Fusion experiments with memory optimization"
            run_experiment "Fusion KMeans RAG" "fusion_kmeans" "$retriever_type" \
                "src/experiments/fusion_kmeans.py"
            cleanup_indices "fusion_kmeans" "$retriever_type"
            
            # Force cleanup between fusion experiments
            cleanup_memory
            
            run_experiment "Fusion Categories RAG" "fusion_categories" "$retriever_type" \
                "src/experiments/fusion_categories.py"
            cleanup_indices "fusion_categories" "$retriever_type"
            ;;
            
        *)
            log_message "ERROR" "Invalid argument. Use --all, --baseline, --kmeans, or --fusion"
            echo "Usage: $0 [--all|--baseline|--kmeans|--fusion] [contriever|bm25|adore|all]"
            exit 1
            ;;
    esac

    # Stop memory monitoring
    kill "$monitor_pid" 2>/dev/null || true

    # Cleanup
    cleanup

    # Deactivate virtual environment if it was activated
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        deactivate
        log_message "INFO" "Virtual environment deactivated"
    fi

    log_message "INFO" "All operations completed successfully!"
}

# Set up trap for cleanup on script exit
trap cleanup EXIT
trap 'kill $(jobs -p) 2>/dev/null' EXIT

# Execute main with all arguments
main "$@"