#!/bin/bash

echo "=================================="
echo "Doublespeak Repository Setup"
echo "=================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Check for GPU
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo ""
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "⚠ No NVIDIA GPU detected - using CPU version"
    echo "  Note: Analysis will be slower on CPU"
fi

# Create output directory
echo ""
echo "Creating output directory..."
mkdir -p outputs

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python3 -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"
python3 -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the complete pipeline:"
echo "  python example_usage.py --model-name meta-llama/Llama-3-8B-Instruct"
echo ""
echo "For help:"
echo "  python example_usage.py --help"
echo ""
