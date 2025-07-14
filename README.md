This project is an email intent classification template using Google's Gemma 2B model. The current version:
- Loads Google's Gemma 2B model
- Runs inference on a small set of example emails (single and batch)
- Integrates the `elenigkove/Email_Intent_Classification` dataset
- Supports LoRA fine-tuning with Hugging Face Trainer and PEFT
- Supports Apple Silicon (MPS) and CPU devices

## Setup
### 1. Create and activate a virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) In VS Code
- Open the Command Palette (Cmd+Shift+P)
- Select `Python: Select Interpreter`
- Choose the `.venv` environment for this workspace

## Run Inference
To run single email inference:
```bash
python3 train.py
```

To run batch inference:
```bash
python3 evaluation_batch_inference.py
```

## Example Output
After running inference, you should see output similar to:

```
Classifying Emails: 100%|██████████| 4/4 [00:10<00:00,  2.50s/it]

=== Email #1 ===
Can you please confirm the meeting time for tomorrow?

Predicted Intent: Request
Time Taken: 2.45 seconds

=== Email #2 ===
We have processed your payment successfully.

Predicted Intent: Transaction
Time Taken: 2.38 seconds

...etc.
```

## Usage for Training
LoRA fine-tuning is supported. To start training:
```bash
python3 -c "from src.trainer import fine_tune_lora; fine_tune_lora()"
```
This will fine-tune the model using the integrated dataset and save the LoRA adapter to `outputs/lora_adapter`.

## Troubleshooting
- **Model download is slow or fails:** Ensure you have a stable internet connection. Try running the script again.
- **CUDA out of memory:** If you are using a GPU and encounter memory errors, try running on CPU or use a smaller model. (This project is optimized for MPS/CPU.)
- **ModuleNotFoundError:** Make sure you have activated your virtual environment and installed all dependencies with `pip install -r requirements.txt`.
- **Other errors:** Check that your Python version is 3.8 or higher. If issues persist, please open an issue or consult the documentation for the relevant library.

## Project Structure
- `src/config.py`: Model and training configuration
- `src/data_loader.py`: Loads and preprocesses the dataset
- `src/model_loader.py`: Loads model and tokenizer
- `src/evaluate.py`: Inference utilities (single and batch)
- `src/trainer.py`: LoRA fine-tuning pipeline
- `train.py`: Single email inference script
- `evaluation_batch_inference.py`: Batch inference script
- `test_*.py`: Test scripts for data and model

## Requirements
See `requirements.txt` for all dependencies. Key packages:
- transformers
- datasets
- torch
- peft
- accelerate
- tqdm

## Next Steps
- Build AI agent integration
