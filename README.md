<a id="readme-top"></a>

<div align="center"> 
    <h1 align="center">Convolutional neural network style transfer analysis</h1> 
</div> 

## Getting Started
Before running the project, ensure you have Python 3.10+ installed.

Check your version:
```bash
python --version
```

###  Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/lazarnagulov/cnn-style-transfer-analysis.git
cd cnn-style-transfer-analysis
```
2. Create and activate virtual environment (optional but recommended):
```bash
chmod +x venv.sh && ./venv.sh   # Linux / macOS
call venv.bat                   # Windows (use CMD, not integrated terminal) 
```
3.  Install dependencies:

#### Option A – CPU-only (any system)
Install shared dependencies and CPU PyTorch:
```bash
pip install -r requirements.txt
pip install -r requirements_cpu.txt
```
#### Option B – GPU / CUDA users

1. Skip installing CPU PyTorch. If previously installed, uninstall it:
```bash
pip uninstall torch torchvision -y
```
2. Install GPU-enabled PyTorch. For example, for CUDA 13.0 / RTX 3060:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
3. Install other dependencies:
```bash
pip install -r requirements.txt
```
4. Verify GPU installation
```python
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))
```
Expected output for a CUDA setup:
```
PyTorch version: 2.9.1+cu130
CUDA available: True
GPU device: NVIDIA GeForce RTX 3060
```
If `CUDA available` is True, your Neural Style Transfer will automatically run on the GPU, dramatically speeding up training.
<br/>