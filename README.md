# IndicIR
This repo includes the model to translate english datasets to malayalam

# IndicTrans2 Dataset Translator

## Quick Start

1. **Clone this repository**


2. **Create and activate the conda environment**


3. **Run the translation script**


- Input: `SciFact/data/data/corpus.jsonl`
- Output: `Translated_datasets/corpus_translated.jsonl`
- Source language: English
- Target language: Malayalam
- Batch size: 32

**Note:** This setup assumes CUDA is already installed and properly configured on your system for GPU acceleration.  
If you do **not** have CUDA installed:

- Install the latest NVIDIA GPU driver for your system: [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- Install the CUDA Toolkit matching the version in `environment.yml`: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- (Optional) Update the CUDA version in `environment.yml` to match your installed version (e.g., change `cudatoolkit=11.8` to your version).
- Recreate the environment:


You’re ready to translate your dataset!
