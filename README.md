# LoRA Fine-tuning Pipeline for Email Intent Classification

A clean, modular pipeline for fine-tuning Google's Gemma 2B model using LoRA (Low-Rank Adaptation) on email intent classification tasks.

## 🏗️ Project Structure

```
├── configs/
│   ├── __init__.py         # Configuration package
│   └── config.py           # All configuration settings
├── src/
│   ├── __init__.py         # Package initialization
│   ├── model_handler.py    # Model operations (load, LoRA, save)
│   ├── data_handler.py     # Dataset operations (load, preprocess)
│   ├── evaluator.py        # Evaluation and metrics
│   ├── trainer.py          # LoRA training pipeline
│   └── data_loader.py      # Legacy data loading utilities
├── main.py                 # Single CLI entry point
├── outputs/               # Training outputs (ignored by git)
├── requirements.txt       # Dependencies
└── README.md             # This file
```## 🚀 Quick Start### 1. Setup Environment```bash# Create virtual environmentpython3 -m venv .venvsource .venv/bin/activate  # On Windows: .venv\Scripts\activate# Install dependenciespip install -r requirements.txt```### 2. Run Complete Pipeline```bash# Run the full pipeline: info → base test → train → lora test → comparepython main.py full```### 3. Individual Commands```bash# Show configuration and dataset infopython main.py info# Test base model performancepython main.py base-test# Train LoRA adapterpython main.py train# Test trained LoRA modelpython main.py lora-test# Compare base vs LoRA modelpython main.py compare# Interactive testing with custom emailspython main.py interactive```## 🎛️ Advanced Usage### Custom Training Parameters```bash# Train with custom settingspython main.py train --epochs 5 --learning-rate 1e-4 --batch-size 2```### Using Different Adapter Path```bash# Test specific adapterpython main.py lora-test --adapter-path /path/to/my/adapter# Compare with specific adapterpython main.py compare --adapter-path outputs/my_custom_adapter```## 📊 Pipeline Workflow### 1. **Information Phase** (`python main.py info`)- Show model and dataset configuration- Display dataset statistics and sample examples- Verify environment setup### 2. **Base Model Testing** (`python main.py base-test`)- Load Google Gemma-2B base model
- Test on predefined email examples
- Establish baseline performance

### 3. **LoRA Training** (`python main.py train`)
- Load and preprocess Email Intent Classification dataset
- Apply LoRA configuration to base model
- Train adapter with configurable parameters
- Save trained adapter to `outputs/lora_adapter/`

### 4. **LoRA Model Testing** (`python main.py lora-test`)
- Load trained LoRA adapter
- Test on same examples as base model
- Calculate accuracy and performance metrics

### 5. **Model Comparison** (`python main.py compare`)
- Automatically compare base vs LoRA model
- Show accuracy improvements and inference time changes
- Generate detailed comparison report

### 6. **Interactive Testing** (`python main.py interactive`)
- Test model with custom email inputs
- Real-time intent classification
- Explore model behavior on new examples

## 🔧 Configuration

All configuration is centralized in `configs/config.py`:

### Model Configuration
- Base model: `google/gemma-2b`
- Max tokens: `128`
- Device: Auto-detected (MPS/CUDA/CPU)

### LoRA Configuration
- Rank: `8` (adjustable for parameter efficiency)
- Alpha: `16` (controls adaptation strength)
- Target modules: `["q_proj", "v_proj"]`
- Dropout: `0.05`

### Training Configuration
- Epochs: `2` (adjustable via CLI)
- Batch size: `1` (adjustable via CLI)
- Learning rate: `2e-4` (adjustable via CLI)
- Evaluation steps: `100`

### Intent Categories
- **Request**: Asking for information or action
- **Informational**: Providing information or updates  
- **Transaction**: Payment or business transaction related
- **Feedback**: Requesting or providing feedback

## 📈 Expected Performance

### Current Results (baseline):
- **Base Model**: ~25% accuracy (essentially random)
- **LoRA Fine-tuned**: ~25-50% accuracy (varies by training)
- **Training Time**: ~2-5 minutes on Apple Silicon
- **Dataset Size**: 20 training samples, 5 test samples

### Optimization Opportunities:
- Increase training epochs (try 5-10)
- Expand LoRA rank (try 16 or 32)
- Add more target modules
- Improve prompt engineering
- Use larger datasets

## 🛠️ Development

### Architecture Benefits:
1. **Modular Design**: Each component has single responsibility
2. **Centralized Config**: All settings in one place
3. **Unified CLI**: Single entry point for all operations
4. **No Code Duplication**: Shared utilities and handlers
5. **Clear Pipeline**: Easy to follow workflow
6. **Extensible**: Easy to add new features or models

### Key Components:

- **ModelHandler**: Manages all model operations (load, LoRA, generation)
- **DataHandler**: Handles dataset loading, preprocessing, and prompts
- **Evaluator**: Provides comprehensive evaluation and metrics
- **LoRATrainer**: Orchestrates the complete training pipeline
- **Config**: Centralized configuration using dataclasses

## 🔍 Troubleshooting

### Common Issues:

**Model Loading Errors:**
- Ensure stable internet for model download
- Check available disk space (models are ~5GB)
- Verify Python version (3.8+)

**Memory Issues:**
- Model runs on CPU with MPS fallback
- Reduce batch size if memory errors occur
- Close other applications to free RAM

**Training Issues:**
- Check dataset download and access
- Ensure output directory is writable
- Monitor training logs for errors

### Debug Commands:

```bash
# Check configuration
python main.py info

# Test environment
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"

# Verify dependencies
pip list | grep -E "(torch|transformers|peft|datasets)"
```

## 📝 Next Steps

### Immediate Improvements:
1. **Increase Training Data**: Use larger email datasets
2. **Hyperparameter Tuning**: Optimize learning rate, epochs, rank
3. **Better Prompting**: Improve instruction templates
4. **Evaluation Metrics**: Add precision, recall, F1-score

### Advanced Features:
1. **Multi-Model Support**: Add support for other base models
2. **Custom Datasets**: Easy integration of new datasets  
3. **Wandb Integration**: Training visualization and tracking
4. **Model Deployment**: API server for production use
5. **Batch Processing**: Handle multiple emails efficiently

## 📄 License

This project is for educational and research purposes. Please respect the licensing terms of the underlying models and datasets.

## 🤝 Contributing

The modular architecture makes it easy to contribute:
1. Fork the repository
2. Create a feature branch
3. Add your improvements to the appropriate module
4. Test with `python main.py full`
5. Submit a pull request

---

**Happy fine-tuning! 🎯**
