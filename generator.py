import torch
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import load_config, load_model_tokenizer, check_cuda

# Configure logging
logging.basicConfig(
    filename="app.log", 
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

# Load configuration with error handling
try:
    config = load_config()
    logging.info("Configuration file loaded successfully.")
except FileNotFoundError:
    logging.error("config.json file not found. Ensure the file exists in the root directory.")
    config = None
except Exception as e:
    logging.error(f"Unexpected error while loading config.json: {e}")
    config = None

# Load model and tokenizer with error handling
if config:
    try:
        model, tokenizer = load_model_tokenizer(config["output_dir"])
        device = check_cuda()
        model.to(device)
        logging.info("Model and tokenizer loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Model directory '{config['output_dir']}' not found.")
        model, tokenizer = None, None
    except Exception as e:
        logging.error(f"Error loading model/tokenizer: {e}")
        model, tokenizer = None
else:
    model, tokenizer = None, None

# Define request model
class TextRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(request: TextRequest):
    """API Endpoint to generate text based on input prompt."""
    if model is None or tokenizer is None:
        logging.error("Model is not loaded properly. Cannot process requests.")
        raise HTTPException(status_code=500, detail="Model not loaded properly. Check logs for details.")
    
    prompt = request.prompt.strip()
    if not prompt:
        logging.warning("Invalid request: Prompt is empty.")
        raise HTTPException(status_code=400, detail="Prompt must be a non-empty string.")
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_length=config["max_length"],
            temperature=config["temperature"],
            top_k=config["top_k"],
            top_p=config["top_p"],
            repetition_penalty=config["repetition_penalty"],
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Generated response: {generated_text}")
        return {"generated_text": generated_text}
    except torch.cuda.OutOfMemoryError:
        logging.error("GPU memory exhausted. Reduce batch size or switch to CPU.")
        raise HTTPException(status_code=500, detail="Out of GPU memory. Try reducing batch size or using CPU.")
    except Exception as e:
        logging.error(f"Internal Server Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    if config:
        try:
            logging.info(f"Starting FastAPI server on {config['host']}:{config['port']}")
            uvicorn.run(app, host=config["host"], port=config["port"])  # Run FastAPI with Uvicorn
        except Exception as e:
            logging.critical(f"Error starting FastAPI server: {e}")
    else:
        logging.critical("Server could not start due to missing or invalid configuration.")