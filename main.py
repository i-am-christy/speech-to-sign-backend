import sys
import os
import torch
import yaml
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel

# Import your ASR engine
from asr_engine import ASREngine

# SETUP PATHS 
CODE_DIR = os.path.join(os.path.dirname(__file__), "ProgressiveTransformersSLP")
sys.path.append(CODE_DIR)

# Global State Variables
nsl_model = None
src_vocab = None
cfg = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Special Tokens (detected automatically on load)
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

# Initialize ASR System Instance
asr_system = ASREngine()

# Try to import NSL modules (safe import)
try:
    from model import build_model
    from data import SRC 
except ImportError:
    print("Warning: Could not import 'model' or 'data'. NSL features may not work.")
    pass 

# --- HELPER CLASS ---
class MockField:
    """
    Wraps the raw vocab object so build_model can access .vocab
    AND len() without crashing on old files.
    """
    def __init__(self, vocab):
        self.vocab = vocab
        # Default attributes expected by some models
        self.init_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN

    # FIX: Count the keys in '.stoi' directly
    # (We bypass len(self.vocab) because it causes the AttributeError)
    def __len__(self):
        return len(self.vocab.stoi)

    # FIX: Safety net for other attributes
    def __getattr__(self, name):
        return getattr(self.vocab, name)

# --- LIFESPAN (Startup & Shutdown Logic) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*30)
    print("STARTING SYSTEM BOOT...")
    
    # 1. Load ASR Model (Whisper)
    try:
        asr_system.load_model()
        print("ASR Model Loaded.")
    except Exception as e:
        print(f"Error loading ASR model: {e}")

    # 2. Load NSL Model (Text -> Sign)
    global nsl_model, src_vocab, cfg, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
    try:
        # Load Config
        config_path = os.path.join(os.path.dirname(__file__), "h2s_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f: 
                cfg = yaml.safe_load(f)
            
            # Load Vocab
            vocab_path = os.path.join(os.path.dirname(__file__), "models", "nsl_vocab.pt")
            if os.path.exists(vocab_path):
                # Load Raw Vocab
                raw_vocab = torch.load(vocab_path, map_location=device, weights_only=False)
                src_vocab = raw_vocab # Keep raw vocab for inference lookup
                
                # Wrap it in MockField
                vocab_wrapper = MockField(raw_vocab)
                
                # Detect Tokens
                stoi = raw_vocab.stoi
                if '<bos>' in stoi: SOS_TOKEN = '<bos>'
                elif '<s>' in stoi: SOS_TOKEN = '<s>'
                
                if '</s>' in stoi: EOS_TOKEN = '</s>'
                
                print(f"   Tokens detected: SOS='{SOS_TOKEN}', EOS='{EOS_TOKEN}'")

                # Update wrapper
                vocab_wrapper.init_token = SOS_TOKEN
                vocab_wrapper.eos_token = EOS_TOKEN
            
                # Build and Load Model
                if cfg:
                    # PASS THE WRAPPER
                    nsl_model = build_model(cfg, vocab_wrapper, None)
                    
                    ckpt_path = os.path.join(os.path.dirname(__file__), "models", "nsl_final.ckpt")
                    if os.path.exists(ckpt_path):
                        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                        nsl_model.load_state_dict(state_dict)
                        nsl_model.to(device)
                        nsl_model.eval()
                        print("NSL Model Ready")
                    else:
                        print(f"Warning: Checkpoint not found at {ckpt_path}")
            else:
                print("NSL Vocab file not found.")
        else:
             print(f"Warning: Config not found at {config_path}")

    except Exception as e:
        print(f"NSL Model Warning: {e}")
        import traceback
        traceback.print_exc()

    print("System Online")
    print("="*30 + "\n")
    
    yield  # Application runs here
    
    print("Shutting down SignBridge System...")

# --- APP INITIALIZATION ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

# --- HELPER FUNCTIONS ---
def run_nsl_inference(text):
    if not nsl_model: 
        print("Error: NSL Model is not loaded.")
        return {"animation": []}
    
    try:
        # Use the detected special tokens
        tokens = [SOS_TOKEN] + text.lower().split() + [EOS_TOKEN]
        
        # Safe lookup for UNK token
        unk_idx = src_vocab.stoi.get(UNK_TOKEN, 0)
        src_idxs = [src_vocab.stoi.get(t, unk_idx) for t in tokens]
        
        src = torch.LongTensor(src_idxs).unsqueeze(0).to(device)
        trg = torch.zeros(1, 1, cfg['model']['input_size']).to(device)
        
        results = []
        with torch.no_grad():
            memory = nsl_model.encode(src, None)
            for i in range(120): # Generate up to 120 frames
                out = nsl_model.decode(memory, None, trg, None)
                pred = nsl_model.generator(out[:, -1, :])
                trg = torch.cat([trg, pred.unsqueeze(1)], dim=1)
                results.append(pred.squeeze().cpu().numpy().tolist())
                
        return {"animation": results}
    except Exception as e:
        print(f"Inference Error: {e}")
        return {"animation": [], "error": str(e)}

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "online", "message": "SignBridge API is running!"}

@app.get("/debug_status")
def debug_status():
    """Checks if the models are actually loaded in memory."""
    global nsl_model, src_vocab, cfg
    
    status = {
        "ASR_System": "Ready" if asr_system else "Not Initialized",
        "NSL_Model_Loaded": nsl_model is not None,
        "Vocab_Loaded": src_vocab is not None,
        "Config_Loaded": cfg is not None,
        "Device": str(device)
    }
    
    base = os.path.dirname(__file__)
    files_check = {
        "Config File": os.path.exists(os.path.join(base, "h2s_config.yaml")),
        "Vocab File": os.path.exists(os.path.join(base, "models", "nsl_vocab.pt")),
        "Checkpoint File": os.path.exists(os.path.join(base, "models", "nsl_final.ckpt")),
    }
    
    return {"model_status": status, "file_system_check": files_check}

@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 1. Convert Audio -> Text 
        print("Transcribing audio...")
        transcribed_text = asr_system.transcribe(temp_filename)
        print(f"Text: {transcribed_text}")

        # 2. Convert Text -> Animation
        if transcribed_text:
            animation_data = run_nsl_inference(transcribed_text)
            return {
                "text": transcribed_text,
                "animation": animation_data.get("animation", [])
            }
        else:
             return {"text": "", "animation": [], "error": "Could not transcribe audio"}
        
    except Exception as e:
        print(f"Endpoint Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)