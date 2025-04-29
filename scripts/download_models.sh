#!/bin/bash

# Script to download LLM and embedding models for the Eureka system
set -e

MODELS_DIR="../models"
mkdir -p $MODELS_DIR

echo "===================================="
echo "Eureka RAG System Model Downloader"
echo "===================================="
echo ""
echo "This script will download models required for the Eureka RAG system."
echo "Models will be stored in the $MODELS_DIR directory."
echo ""

# Function to download with progress using wget or curl
download_file() {
    URL=$1
    OUTPUT=$2

    if command -v wget > /dev/null; then
        wget --progress=bar:force -O "$OUTPUT" "$URL"
    elif command -v curl > /dev/null; then
        curl -L --progress-bar -o "$OUTPUT" "$URL"
    else
        echo "Error: Neither wget nor curl is installed. Please install one of them and try again."
        exit 1
    fi
}

# LLM Model Selection
echo "Select an LLM model to download:"
echo "1) Llama-3-8B-Instruct (Q4_K_M - 4.8GB)"
echo "2) Mistral-7B-Instruct (Q4_K_M - 4.5GB)"
echo "3) Phi-2 (Q4_K_M - 2.8GB)"
echo "4) Skip LLM download (if you already have a model)"
echo ""
read -p "Enter choice [1-4]: " llm_choice

case $llm_choice in
    1)
        MODEL_NAME="llama-3-8b-instruct.Q4_K_M.gguf"
        MODEL_URL="https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf"
        ;;
    2)
        MODEL_NAME="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        ;;
    3)
        MODEL_NAME="phi-2.Q4_K_M.gguf"
        MODEL_URL="https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
        ;;
    4)
        echo "Skipping LLM model download."
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Download LLM model if not skipped
if [ $llm_choice -ne 4 ]; then
    echo ""
    echo "Downloading $MODEL_NAME..."
    download_file "$MODEL_URL" "$MODELS_DIR/$MODEL_NAME"
    echo "LLM model downloaded successfully to $MODELS_DIR/$MODEL_NAME"
fi

# For embedding model, we'll use sentence-transformers which will be auto-downloaded
# by the Python code, but we can inform the user
echo ""
echo "The embedding model (sentence-transformers) will be downloaded automatically"
echo "when the Vector DB service starts for the first time."
echo ""

# Complete
echo "===================================="
echo "Model setup completed!"
echo ""
echo "You can now update your .env file with:"
echo "LLM_MODEL_FILE=$MODEL_NAME"
echo ""
echo "Make sure to set all other required environment variables in the .env file."
echo "===================================="
