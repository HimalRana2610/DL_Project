# Deep Learning Project: Prompt Generation + CLIP Training

This project builds text prompts from ISGD facial attributes, uses LLMs to generate natural-language descriptions, and fine-tunes CLIP for text-to-image matching.

## Project Overview

The workflow has two stages:

1. Prompt generation
- Reads facial attributes from `ISGD/Attributes.csv`
- Builds rule-based prompts
- Generates LLM prompts using Llama / Mistral / Gemma APIs
- Exports merged prompt dataset to `isgd_prompts.csv`

2. CLIP training and retrieval
- Reads `isgd_prompts.csv`
- Builds `clip_dataset.json` with image-path + text pairs
- Fine-tunes `openai/clip-vit-base-patch32`
- Runs text-to-image retrieval and shows top matches

## Repository Structure

- `prompt_generator.ipynb`: Prompt generation pipeline
- `clip_training.ipynb`: CLIP dataset build, training, and retrieval
- `ISGD/Attributes.csv`: Attribute labels
- `ISGD/Images/`: Face images
- `isgd_prompts.csv`: Final prompts dataset used for CLIP training
- `clip_dataset.json`: Intermediate CLIP training dataset
- `clip_finetuned/`: Saved fine-tuned CLIP model + processor
- `temp_llama.csv`, `temp_mistral.csv`, `temp_gemma.csv`: Checkpoint CSV files during generation
- `requirements.txt`: Python dependencies snapshot

## Setup

### 1. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& ".\\.venv\\Scripts\\Activate.ps1")
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

If dependency installation fails due encoding issues in `requirements.txt`, regenerate it in UTF-8:

```powershell
pip freeze | Out-File -Encoding utf8 requirements.txt
```

### 3. Download the dataset from Kaggle

Dataset: https://www.kaggle.com/datasets/himalrana2610/indian-skincare-and-grooming-dataset

1. Install Kaggle CLI (if not already installed):

```powershell
pip install kaggle
```

2. Add your Kaggle API token:
- Go to Kaggle Account -> API -> Create New Token.
- Place `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json`.

3. Download and extract the dataset in the project root:

```powershell
kaggle datasets download -d himalrana2610/indian-skincare-and-grooming-dataset
Expand-Archive -Path .\indian-skincare-and-grooming-dataset.zip -DestinationPath . -Force
```

4. Ensure extracted data is available as:
- `ISGD/Attributes.csv`
- `ISGD/Images/`

### 4. Launch Jupyter

```powershell
jupyter notebook
```

## Run Order

Run notebooks in this order:

1. `prompt_generator.ipynb`
- Produces LLM-based prompts for a subset of images (currently first 100 rows)
- Writes `isgd_prompts.csv`

2. `clip_training.ipynb`
- Builds `clip_dataset.json`
- Fine-tunes CLIP
- Saves model to `clip_finetuned/`
- Runs retrieval and displays matching images

## Outputs

- Prompt files
  - `temp_llama.csv`
  - `temp_mistral.csv`
  - `temp_gemma.csv`
  - `isgd_prompts.csv`

- CLIP artifacts
  - `clip_dataset.json`
  - `clip_finetuned/`

## Troubleshooting

- `ERROR` prompt rows in Together models:
  - The notebook includes fallback to Groq Llama when Together returns `402` (credit limit).

- DataLoader tensor-size mismatch in CLIP training:
  - Resolved using batched tokenization in a `collate_fn`.

- Retrieval errors (`input_ids` / `pixel_values` missing):
  - Resolved by using CLIP text and vision submodules for standalone embeddings.

## Notes

- API keys should be moved to environment variables before sharing this project.
- Current prompt-generation notebook is configured for a fast 100-image run for iteration speed.
