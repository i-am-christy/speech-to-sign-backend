import os
import whisper
import torch

class ASREngine:
    def __init__(self):
        self.model = None

    def load_model(self):
        # Define the specific path for the model
        model_path = os.path.join(os.getcwd(), "models", "asr_model")
        
        # Check if the folder exists, if not, create it
        if not os.path.exists(model_path):
            print(f"Directory {model_path} not found. Creating it...")
            os.makedirs(model_path, exist_ok=True)

        print(f"Loading Whisper Model...")
        
        try:
            # 1. Try loading from the local path first
            if os.path.exists(os.path.join(model_path, "model.pt")):
                print(f"Found local model at {model_path}")
                self.model = whisper.load_model(model_path)

            # 2. If local fails or doesn't exist, download 'base' model
            else:
                print("Local model not found. Downloading 'base' model...")
                self.model = whisper.load_model("base") 
                
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to downloading base model directly if path logic fails
            self.model = whisper.load_model("base")

        print("Whisper Model Loaded Successfully.")

    def transcribe(self, audio_input):
        """
        Transcribes speech to text.
        """
        if self.model is None:
            print("Error: Model not loaded.")
            return ""
            
        try:
            # Whisper can take a file path or a numpy array
            result = self.model.transcribe(audio_input)
            return result["text"].strip()
        except Exception as e:
            print(f"Transcription Error: {e}")
            return ""