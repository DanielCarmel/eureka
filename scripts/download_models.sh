#!/bin/bash

# Script to download LLM and embedding models for the Eureka system
set -e

MODELS_DIR="models"
EMBEDDINGS_DIR="$MODELS_DIR/embeddings"
mkdir -p $MODELS_DIR
mkdir -p $EMBEDDINGS_DIR

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
        MODEL_URL="https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
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

# Embedding model download option
echo ""
echo "Do you want to download the embedding model for offline use?"
echo "1) Yes, download all-MiniLM-L6-v2 (80MB)"
echo "2) Skip embedding model download"
echo ""
read -p "Enter choice [1-2]: " embedding_choice

if [ "$embedding_choice" -eq 1 ]; then
    # Create a temporary Python script to download the model
    TEMP_SCRIPT=$(mktemp)
    cat > $TEMP_SCRIPT << 'EOF'
from sentence_transformers import SentenceTransformer
import os
import sys

# Get the cache directory from arguments or use default
cache_dir = sys.argv[1] if len(sys.argv) > 1 else "models/embeddings"
model_name = "all-MiniLM-L6-v2"

print(f"Downloading embedding model {model_name} to {cache_dir}...")
try:
    # This will download and cache the model
    model = SentenceTransformer(model_name, cache_folder=cache_dir)
    print(f"Successfully downloaded and cached model to {cache_dir}")
except Exception as e:
    print(f"Error downloading model: {e}")
    sys.exit(1)
EOF

    # Check if Python and pip are installed
    if command -v python3 > /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python > /dev/null; then
        PYTHON_CMD="python"
    else
        echo "Error: Python is not installed. Please install Python and try again."
        rm $TEMP_SCRIPT
        exit 1
    fi
    
    # Install sentence-transformers if not installed
    $PYTHON_CMD -c "import sentence_transformers" 2>/dev/null || {
        echo "Installing sentence-transformers package..."
        if command -v pip3 > /dev/null; then
            pip3 install sentence-transformers
        elif command -v pip > /dev/null; then
            pip install sentence-transformers
        else
            echo "Error: pip is not available. Please install pip and try again."
            rm $TEMP_SCRIPT
            exit 1
        fi
    }
    
    # Run the script to download the model
    $PYTHON_CMD $TEMP_SCRIPT "$EMBEDDINGS_DIR"
    RM_STATUS=$?
    
    # Clean up
    rm $TEMP_SCRIPT
    
    if [ $RM_STATUS -ne 0 ]; then
        echo "Failed to download embedding model. Please check your internet connection."
        exit 1
    fi
    
    echo "Embedding model downloaded successfully to $EMBEDDINGS_DIR"
else
    echo "Skipping embedding model download."
    echo "Note: The embedding model will be downloaded automatically when needed if internet is available."
fi

# Complete
echo ""
echo "===================================="
echo "Model setup completed!"
echo ""
echo "You can now update your .env file with:"
echo "LLM_MODEL_FILE=$MODEL_NAME"
echo "MODEL_CACHE_DIR=$EMBEDDINGS_DIR"
echo ""
echo "Make sure to set all other required environment variables in the .env file."
echo "===================================="
