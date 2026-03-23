"""
Fast Dependency Checker for Q1 Whisper Fine-tuning
==================================================
Lightweight version that avoids hanging on scipy/transformers imports.
"""

import sys
import subprocess

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_header(text):
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}{text}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

def print_ok(text):
    print(f"{GREEN}✓{RESET} {text}")

def print_fail(text):
    print(f"{RED}✗{RESET} {text}")

def print_warn(text):
    print(f"{YELLOW}⚠{RESET} {text}")

def main():
    print(f"\n{BOLD}Whisper Fine-tuning - Fast Dependency Checker (v2){RESET}")
    print(f"Python: {sys.executable}\n")
    
    print_header("1. Python Version")
    version_info = sys.version_info
    version_string = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    if version_info >= (3, 8):
        print_ok(f"Python {version_string}")
    else:
        print_fail(f"Python {version_string} (requires 3.8+)")
    
    print_header("2. PyTorch & GPU Status")
    try:
        import torch
        print_ok(f"torch (v{torch.__version__})")
        
        if "cuda" in torch.__version__ or "cu" in torch.__version__:
            print_warn("PyTorch has CUDA in version string, but checking actual GPU...")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print_ok(f"✓ GPU AVAILABLE: {gpu_count} device(s) - {gpu_name}")
            print(f"  → Fine-tuning will use GPU acceleration (fast!)")
        else:
            print_warn(f"✗ GPU NOT AVAILABLE - CPU mode will be very slow")
            print(f"  → This is expected on most machines without CUDA GPU")
            print(f"  → Use --cpu_mode flag for CPU testing")
    except ImportError as e:
        print_fail(f"torch import failed: {e}")
        return 1
    
    print_header("3. Core ML Libraries")
    
    packages = [
        ("transformers", None, (4, 36)),
        ("datasets", None, (2, 14)),
        ("evaluate", None, (0, 4)),
        ("librosa", None, None),
        ("soundfile", None, None),
    ]
    
    all_ok = True
    for pkg_name, import_as, min_version in packages:
        try:
            mod = __import__(pkg_name)
            version = getattr(mod, "__version__", "unknown")
            
            if min_version:
                try:
                    parsed = tuple(int(x) for x in str(version).split(".")[:2])
                    if parsed >= min_version:
                        print_ok(f"{pkg_name} (v{version})")
                    else:
                        min_str = ".".join(str(x) for x in min_version)
                        print_warn(f"{pkg_name} (v{version}) - old version, want {min_str}+")
                except:
                    print_ok(f"{pkg_name} (v{version})")
            else:
                print_ok(f"{pkg_name} (v{version})")
        except ImportError:
            print_fail(f"{pkg_name} - NOT INSTALLED")
            all_ok = False
    
    print_header("4. Optional Dependencies")
    
    opt_packages = ["requests", "accelerate"]
    for pkg_name in opt_packages:
        try:
            mod = __import__(pkg_name)
            version = getattr(mod, "__version__", "unknown")
            print_ok(f"{pkg_name} (v{version})")
        except ImportError:
            print_warn(f"{pkg_name} - optional, can install later")
    
    print_header("5. Dataset Manifest")
    
    import os
    from pathlib import Path
    
    manifest_path = Path(".") / "manifest.json"
    if manifest_path.exists():
        print_ok(f"manifest.json found")
        try:
            import json
            with open(manifest_path) as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                print_ok(f"  {len(data)} entries in manifest")
            else:
                print_warn(f"  Manifest empty or malformed")
        except:
            print_warn(f"  Could not read manifest")
    else:
        print_warn(f"manifest.json NOT FOUND - create one from sample_manifest.json")
    
    sample_manifest = Path(".") / "sample_manifest.json"
    if sample_manifest.exists():
        print_ok(f"sample_manifest.json available (template)")
    
    print_header("6. Disk Space")
    
    try:
        import shutil
        usage = shutil.disk_usage("/")
        free_gb = usage.free / (1024 ** 3)
        
        if free_gb > 50:
            print_ok(f"Disk space: {free_gb:.1f} GB free (sufficient)")
        elif free_gb > 10:
            print_warn(f"Disk space: {free_gb:.1f} GB free (may be tight)")
        else:
            print_fail(f"Disk space: {free_gb:.1f} GB free (INSUFFICIENT)")
    except:
        pass
    
    print_header("Summary & Next Steps")
    
    if all_ok:
        print(f"\n{GREEN}{BOLD}✓ All required packages are installed!{RESET}")
    else:
        print(f"\n{RED}{BOLD}✗ Some packages are missing. Install them:{RESET}")
        print("\n  pip install transformers datasets evaluate librosa soundfile")
        print("  pip install requests accelerate")
    
    print("\n" + "="*60)
    print("READY TO RUN:")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"\n  GPU MODE (recommended for your system):")
        print(f"    python q1_whisper_finetune.py --manifest manifest.json")
        print(f"\n  Optional: Test with sample data first")
        print(f"    python q1_whisper_finetune.py --manifest sample_manifest.json \\")
        print(f"      --sample_size 3 --audio_dir ./test_audio")
    else:
        print(f"\n  CPU MODE (slow, for testing only):")
        print(f"    python q1_whisper_finetune.py --manifest manifest.json --cpu_mode")
        print(f"\n  Better alternative: Use Google Colab (free GPU)")
        print(f"    1. Upload notebook to https://colab.research.google.com")
        print(f"    2. Runtime → Change runtime type → GPU")
        print(f"    3. Run cells (much faster than local CPU!)")
    
    print("\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
