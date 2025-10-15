# GNN-SurgVQA: Object-Centric Graph Reasoning for Visual Question Answering in Laparoscopic Scene Understanding

## 1. Environment Setup and Installation 
To ensure reproducibility and facilitate deployment, the GNN-SurgVQA framework is distributed with environment configuration files for both demonstration and training purposes. The following steps outline the recommended procedure for setting up the computational environment and installing all necessary dependencies:

### 1.1. Prerequisites

- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS (limited support)
- **Python Version**: 3.9 or higher
- **Conda**: Anaconda or Miniconda (recommended for environment management)
- **GPU**: NVIDIA GPU with CUDA 11.8+ (recommended for optimal performance)

### 1.2. Environment Creation

The project provides two environment files:

- `environment_demo.yml`: For running the web-based demonstration interface
- `environment.yml`: For model training and development

To create and activate the environment for the demo, execute the following commands in a terminal:

```bash
conda env create -f environment_demo.yml
conda activate surgical-demo
```

For training and development, use:

```bash
conda env create -f environment.yml
conda activate surgical-vqa
```

### 1.3. Dependency Installation

All required Python packages, including PyTorch, PyTorch Geometric, Ultralytics YOLOv8, Hugging Face Transformers, Gradio, OpenCV, and Pillow, will be installed automatically via the provided environment files. Manual installation is not required unless custom modifications are made.

### 1.4. Verification

After activating the environment, verify the installation by checking the versions of the main libraries:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); import gradio; print('Gradio:', gradio.__version__)"
```

The expected output should confirm the presence of PyTorch (>=2.8.0) and Gradio (>=4.20.0).

### 1.5. Running the Demo

To launch the web-based demonstration interface, execute:

```bash
python surgical_vqa_demo.py
```

Upon successful launch, the interface will be accessible at `http://localhost:7860` in a web browser.

---