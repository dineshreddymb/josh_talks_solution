## Whisper Hindi Fine-Tuning Pipeline - Setup & Usage Guide

### ✓ Installation Status

All required packages are now installed:
```
✓ Python 3.10.11
✓ PyTorch 2.10.0 (CPU-only mode)
✓ transformers 5.3.0
✓ datasets 4.8.3
✓ evaluate 0.4.6
✓ librosa 0.11.0
✓ soundfile 0.13.1
✓ accelerate 1.13.0
✓ requests 2.32.5
```

**Note**: GPU (CUDA) is NOT available on this machine. Training will be in CPU mode (~10-50x slower than GPU). For faster training, consider using Google Colab (free GPU).

---

### 📋 Files Overview

| File | Purpose |
|------|---------|
| `q1_whisper_finetune.py` | **Main script** - Whisper fine-tuning pipeline (MODIFIED for CPU mode) |
| `check_dependencies_fast.py` | Diagnostic tool to verify all packages installed |
| `install_dependencies.py` | Helper script for package installation |
| `sample_manifest.json` | Template manifest with 3 example entries |
| `manifest.json` | Your actual dataset (create from your data) |

---

### 🚀 Quick Start

#### Step 1: Prepare Your Manifest

Create `manifest.json` with your Hindi ASR dataset entries. Each entry should have:

```json
[
  {
    "user_id": "YOUR_USER_ID",
    "recording_id": "REC_ID_123",
    "rec_url_gcp": "https://storage.googleapis.com/upload_goai/YOUR_USER_ID/REC_ID_123.wav",
    "transcription_url": "https://storage.googleapis.com/upload_goai/YOUR_USER_ID/REC_ID_123_transcription.json",
    "metadata_url": "https://storage.googleapis.com/upload_goai/YOUR_USER_ID/REC_ID_123_metadata.json",
    "duration": 10.5,
    "language": "hi"
  },
  ...
]
```

Or start with the sample:
```bash
copy sample_manifest.json manifest.json
```

---

#### Step 2: Run the Fine-tuning Pipeline

**CPU Mode (slow, for testing):**
```bash
python q1_whisper_finetune.py --manifest manifest.json --cpu_mode
```

**With Sample Data (for quick validation):**
```bash
python q1_whisper_finetune.py --manifest sample_manifest.json --cpu_mode --sample_size 3
```

**Full GPU Mode (if GPU becomes available):**
```bash
python q1_whisper_finetune.py --manifest manifest.json
```

---

### 📊 Command-Line Options

```
q1_whisper_finetune.py:

  --manifest PATH           Path to your manifest.json (REQUIRED)
  --audio_dir PATH          Where to cache downloaded audio files (default: ./audio)
  --output PATH             Where to save fine-tuned model (default: ./whisper-hi-finetuned)
  --cpu_mode                Enable CPU mode (disables fp16, reduces batch size & training steps)
  --sample_size N           For testing: use only first N samples from manifest
  --eval_only               Skip training; only run evaluation on FLEURS dataset
```

---

### 🔬 Example Workflows

**Workflow 1: Quick Test (2 min)**
```bash
# Test pipeline with 3 samples, CPU mode
python q1_whisper_finetune.py --manifest sample_manifest.json --cpu_mode --sample_size 3
```

**Workflow 2: Small Training Run (1-2 hours on CPU)**
```bash
# Full pipeline with small dataset sample
python q1_whisper_finetune.py --manifest manifest.json --cpu_mode --sample_size 50
```

**Workflow 3: Full Training (on GPU - much faster)**
```bash
# On machine with GPU or Google Colab:
python q1_whisper_finetune.py --manifest manifest.json
```

**Workflow 4: Evaluation Only**
```bash
# Skip training, just evaluate existing model
python q1_whisper_finetune.py --manifest manifest.json --eval_only
```

---

### 💾 Output Files

After running the pipeline, you'll get:

```
./                                    # Current directory
├── whisper-hi-finetuned/            # Fine-tuned model checkpoint
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   ├── tokenizer_config.json
│   └── ...
├── audio/                            # Cached downloaded audio files
│   ├── recording_id_1.wav
│   ├── recording_id_2.wav
│   └── ...
├── runs/                             # TensorBoard logs
│   └── DATE_TIME/
│       └── events.out...
└── error_analysis.json               # Error taxonomy & sampled errors
```

---

### 📈 What The Pipeline Does

#### 1. **Preprocessing** (~5-10 min for 10h data)
   - Downloads audio + transcriptions from GCP URLs
   - Cleans Hindi text (remove noise, normalize Unicode)
   - Validates audio duration (0.5s - 30s)
   - Splits into 90% train / 10% eval

#### 2. **Fine-tuning** (~30 min - 2 hours on GPU / many hours on CPU)
   - Loads pretrained Whisper-small
   - Trains on Hindi ASR data with:
     - Batch size: 16 (GPU) / 4 (CPU)
     - Learning rate: 1e-5 (conservative to prevent catastrophic forgetting)
     - Max steps: 4000 (GPU) / 100 (CPU)
     - Evaluation every 500 steps (GPU) / 10 steps (CPU)

#### 3. **Evaluation** (~5-10 min)
   - Tests baseline Whisper-small on FLEURS Hindi test set
   - Tests fine-tuned model on same FLEURS test set
   - Reports Word Error Rate (WER) comparison

#### 4. **Error Analysis**
   - Systematically samples 25+ model errors  
   - Builds error taxonomy from data (not pre-defined)
   - Output: `error_analysis.json`

---

### 🐛 Troubleshooting

#### Issue: "ModuleNotFoundError: No module named 'transformers'"
**Solution**: Reinstall dependencies
```bash
pip install --upgrade transformers datasets evaluate librosa soundfile
```

#### Issue: "No CUDA device found"
**Expected behavior** on CPU-only machine. Use `--cpu_mode` flag.
**To use GPU**: Google Colab (free) or AWS EC2 (paid)

#### Issue: Network timeout downloading audio
**Solution**: Use smaller `--sample_size` to test pipeline
```bash
python q1_whisper_finetune.py --manifest manifest.json --cpu_mode --sample_size 5
```

#### Issue: Out of memory errors on CPU
**Solution**: Reduce batch size further in code, or use `--sample_size`

#### Issue: FLEURS dataset download is very slow
**Expected**: First download can take 10+ minutes (downloading audio files)
**Subsequent runs**: Will use cached data (much faster)

---

### 📝 Key Modifications From Original

1. **CPU Mode Support**
   - Auto-detects GPU availability
   - Disables fp16 (mixed precision) on CPU
   - Disables gradient_checkpointing on CPU
   - Reduces batch_size from 16→4 on CPU
   - Reduces max_steps from 4000→100 on CPU

2. **Better Error Handling**
   - Validates manifest.json exists and is valid JSON
   - Falls back to sample_manifest if needed
   - Clear error messages for missing files

3. **Flexible Testing**
   - `--sample_size` flag to test with subset of data
   - Use `sample_manifest.json` for quick validation
   - `--cpu_mode` for testing on CPU machines

4. **Manifest Validation**
   - Checks manifest file format before processing
   - Helps catch configuration issues early

---

### 🎯 Next Steps

1. **Prepare your manifest.json** with real dataset URLs
2. **Test with sample**: `python q1_whisper_finetune.py --manifest sample_manifest.json --cpu_mode --sample_size 3`
3. **For faster training**: Use Google Colab → Runtime → GPU
4. **Run full pipeline**: `python q1_whisper_finetune.py --manifest manifest.json`
5. **Analyze errors**: Check `error_analysis.json` for taxonomies

---

### ⚡ Performance Notes

| Setup | Speed | Cost | Notes |
|-------|-------|------|-------|
| Local CPU (this machine) | ~1x (slowest) | Free | OK for testing only |
| Google Colab GPU | ~20x faster | Free | Recommended for training |
| AWS GPU (p3.2xlarge) | ~20-30x faster | $3-4/hr | For production use |

---

### 📚 References

- [Whisper Model Card](https://huggingface.co/openai/whisper-small)
- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers/)
- [FLEURS Dataset](https://huggingface.co/datasets/google/fleurs)
- [Google Colab (Free GPU)](https://colab.research.google.com/)

---

**Last Updated**: March 23, 2026  
**Python**: 3.10.11  
**Status**: ✓ Ready to train!  

