import sys
import os
import torch

# Setup Path
CODE_DIR = os.path.join(os.path.dirname(__file__), "ProgressiveTransformersSLP")
sys.path.append(CODE_DIR)

print("Testing Imports ---")
try:
    from data import SRC
    from model import build_model
    print("Imports successful: SRC and build_model found.")
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    print("   HINT: You might be missing 'torchtext'. Try running: pip install torchtext")
    sys.exit(1)
except Exception as e:
    print(f"UNEXPECTED IMPORT ERROR: {e}")
    sys.exit(1)

print("\nTesting Vocab Load ---")
vocab_path = os.path.join(os.path.dirname(__file__), "models", "nsl_vocab.pt")
try:
    # Try loading with default settings first
    print(f"Attempting to load: {vocab_path}")
    vocab = torch.load(vocab_path, map_location="cpu")
    print("Vocab loaded successfully!")
except Exception as e:
    print(f"VOCAB LOAD FAILED: {e}")
    if "weights_only" in str(e):
        print("Trying fix for weights_only...")
        try:
            vocab = torch.load(vocab_path, map_location="cpu", weights_only=False)
            print("Vocab loaded successfully with weights_only=False!")
        except Exception as e2:
            print(f"STILL FAILED: {e2}")

print("\n--- 3. Testing Checkpoint Load ---")
ckpt_path = os.path.join(os.path.dirname(__file__), "models", "nsl_final.ckpt")
try:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("Checkpoint loaded successfully!")
except Exception as e:
    print(f"CHECKPOINT FAILED: {e}")