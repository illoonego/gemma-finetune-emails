This is a basic LLaMA 3 based email intent classification project template. The current version:
- Loads Meta's LLaMA 3 7B Instruct model
- Runs inference on a small set of example emails

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
```bash
python3 train.py
```

## Example Output
After running inference, you should see output similar to:

```
Classifying Emails: 100%|██████████| 4/4 [00:10<00:00,  2.50s/it]

=== Email #1 ===
Can you please confirm the meeting time for tomorrow?

Predicted Intent: Meeting Confirmation
Time Taken: 2.45 seconds

=== Email #2 ===
We have processed your payment successfully.

Predicted Intent: Payment Confirmation
Time Taken: 2.38 seconds

...etc.
```

## Usage for Training (if implemented)
Currently, training is not implemented. When training support is added, usage instructions will be provided here.

## Troubleshooting
- **Model download is slow or fails:** Ensure you have a stable internet connection. Try running the script again.
- **CUDA out of memory:** If you are using a GPU and encounter memory errors, try running on CPU or use a smaller model.
- **ModuleNotFoundError:** Make sure you have activated your virtual environment and installed all dependencies with `pip install -r requirements.txt`.
- **Other errors:** Check that your Python version is 3.8 or higher. If issues persist, please open an issue or consult the documentation for the relevant library.

## Next Steps
- Integrate the `elenigkove/Email_Intent_Classification` dataset
- Add LoRA fine-tuning support
- Build AI agent integration
