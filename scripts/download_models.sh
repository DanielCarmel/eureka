#!/bin/bash

# download_models.sh
# Script to download GGUF model files for local LLM usage with llama.cpp

set -e  # Exit on error

# Default values
DEFAULT_MODEL="llama-2-7b-chat"
DEFAULT_QUANTIZATION="Q4_0"
MODELS_DIR="models"

# Color formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print usage information
usage() {
    echo -e "${BLUE}Download GGUF Model Files${NC}"
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -m, --model NAME       Model name to download (default: $DEFAULT_MODEL)"
    echo "  -q, --quantization Q   Quantization level (default: $DEFAULT_QUANTIZATION)"
    echo "  -d, --dir DIRECTORY    Download directory (default: $MODELS_DIR)"
    echo "  -l, --list             List available models"
    echo "  -h, --help             Show this help message"
    echo
    echo "Available models (use with -m):"
    echo "  llama-2-7b-chat        Meta's Llama 2 7B chat model"
    echo "  llama-2-13b-chat       Meta's Llama 2 13B chat model"
    echo "  mistral-7b-instruct    Mistral 7B Instruct model"
    echo "  mixtral-8x7b-instruct  Mixtral 8x7B Instruct model"
    echo "  falcon-7b-instruct     Falcon 7B Instruct model"
    echo
    echo "Quantization options (use with -q):"
    echo "  Q4_0    4-bit quantization, best compromise size/quality (default)"
    echo "  Q5_K    5-bit quantization, higher quality but larger files"
    echo "  Q8_0    8-bit quantization, highest quality but largest files"
    echo "  Q2_K    2-bit quantization, smallest files but lower quality"
    echo
    echo "Example: $0 -m mistral-7b-instruct -q Q4_0 -d ./my_models"
}

# List available models with descriptions
list_models() {
    echo -e "${BLUE}Available Models:${NC}"
    echo -e "${GREEN}llama-2-7b-chat${NC} - Meta's Llama 2 7B chat model (4.1GB Q4_0)"
    echo "  - Good for general chat and knowledge tasks"
    echo "  - Source: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF"
    echo
    echo -e "${GREEN}llama-2-13b-chat${NC} - Meta's Llama 2 13B chat model (7.3GB Q4_0)"
    echo "  - Better reasoning than 7B version"
    echo "  - Source: https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF"
    echo
    echo -e "${GREEN}mistral-7b-instruct${NC} - Mistral 7B Instruct model (4.1GB Q4_0)"
    echo "  - High performance model, often better than Llama 2 7B"
    echo "  - Source: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    echo
    echo -e "${GREEN}mixtral-8x7b-instruct${NC} - Mixtral 8x7B Instruct model (12.9GB Q4_0)"
    echo "  - Mixture of Experts model, high performance"
    echo "  - Source: https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
    echo
    echo -e "${GREEN}falcon-7b-instruct${NC} - Falcon 7B Instruct model (4.1GB Q4_0)"
    echo "  - Open source alternative"
    echo "  - Source: https://huggingface.co/TheBloke/falcon-7b-instruct-GGUF"
    echo
}

# Download a model
download_model() {
    local model=$1
    local quant=$2
    local dir=$3
    local url=""
    local filename=""

    # Ensure the download directory exists
    mkdir -p "$dir"

    # Define the URL based on the model name
    case $model in
        "llama-2-7b-chat")
            url="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.$quant.gguf"
            filename="llama-2-7b-chat.$quant.gguf"
            ;;
        "llama-2-13b-chat")
            url="https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.$quant.gguf"
            filename="llama-2-13b-chat.$quant.gguf"
            ;;
        "mistral-7b-instruct")
            url="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.$quant.gguf"
            filename="mistral-7b-instruct-v0.2.$quant.gguf"
            ;;
        "mixtral-8x7b-instruct")
            url="https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.$quant.gguf"
            filename="mixtral-8x7b-instruct-v0.1.$quant.gguf"
            ;;
        "falcon-7b-instruct")
            url="https://huggingface.co/TheBloke/falcon-7b-instruct-GGUF/resolve/main/falcon-7b-instruct.$quant.gguf"
            filename="falcon-7b-instruct.$quant.gguf"
            ;;
        *)
            echo -e "${RED}Error: Unsupported model '$model'${NC}"
            exit 1
            ;;
    esac

    echo -e "${BLUE}Downloading model: ${GREEN}$model${NC} with ${GREEN}$quant${NC} quantization..."
    echo -e "${YELLOW}This may take a while depending on your internet connection.${NC}"
    echo -e "Downloading to: ${GREEN}$dir/$filename${NC}"

    # Check if wget or curl is available
    if command -v wget &> /dev/null; then
        wget -c "$url" -O "$dir/$filename"
    elif command -v curl &> /dev/null; then
        curl -L "$url" -o "$dir/$filename" --progress-bar
    else
        echo -e "${RED}Error: Neither wget nor curl is installed. Please install one of them and try again.${NC}"
        exit 1
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Download completed successfully!${NC}"
        echo -e "You can use this model with: ${YELLOW}python main.py --provider llamacpp --model-path $dir/$filename${NC}"
    else
        echo -e "${RED}Download failed. Please check your internet connection and try again.${NC}"
        exit 1
    fi
}

# Parse command-line arguments
MODEL=$DEFAULT_MODEL
QUANTIZATION=$DEFAULT_QUANTIZATION
DIR=$MODELS_DIR

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift
            shift
            ;;
        -q|--quantization)
            QUANTIZATION="$2"
            shift
            shift
            ;;
        -d|--dir)
            DIR="$2"
            shift
            shift
            ;;
        -l|--list)
            list_models
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Execute the download
download_model "$MODEL" "$QUANTIZATION" "$DIR"

echo -e "${BLUE}===============================================${NC}"
echo -e "${GREEN}Model downloaded and ready to use!${NC}"
echo -e "${BLUE}===============================================${NC}"
echo -e "Run with: ${YELLOW}python main.py --provider llamacpp --model-path $DIR/$(ls -t $DIR | head -1)${NC}"
