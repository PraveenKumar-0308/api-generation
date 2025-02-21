import os
import json
import torch
import logging
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_config(config_path="config.json"):
    """Loads configuration from JSON file."""
    try:
        if not os.path.exists(config_path):
            logging.error(f"Configuration file '{config_path}' not found.")
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

        with open(config_path, "r") as file:
            config = json.load(file)
            logging.info(f"Configuration loaded successfully from {config_path}.")
            return config

    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in '{config_path}'. Check for syntax errors.")
        raise ValueError(f"Error: Invalid JSON format in '{config_path}'. Check for syntax errors.")
    except Exception as e:
        logging.error(f"Unexpected error while loading config.json: {e}")
        raise RuntimeError(f"Unexpected error while loading config.json: {e}")

def load_model_tokenizer(model_path):
    """Loads GPT-2 model and tokenizer from a given path."""
    try:
        if not os.path.exists(model_path):
            logging.error(f"Model path '{model_path}' not found.")
            raise FileNotFoundError(f"Model path '{model_path}' not found.")

        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
        # Set pad_token to eos_token if not defined
        tokenizer.pad_token = tokenizer.eos_token

        logging.info(f"Model and tokenizer loaded successfully from {model_path}.")
        return model, tokenizer

    except FileNotFoundError as e:
        logging.error(f"File Error: {e}")
        raise FileNotFoundError(f"File Error: {e}")
    except ValueError as e:
        logging.error(f"Value Error: {e}")
        raise ValueError(f"Value Error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while loading model/tokenizer: {e}")
        raise RuntimeError(f"Unexpected error while loading model/tokenizer: {e}")

def check_cuda():
    """Checks for CUDA availability and returns the appropriate device."""
    try:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logging.info(f"CUDA detected: {torch.cuda.is_available()}, using device: {device}")
        return device
    except Exception as e:
        logging.error(f"Error detecting CUDA: {e}")
        raise RuntimeError(f"Error detecting CUDA: {e}")
