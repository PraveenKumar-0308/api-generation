import os
import torch
import json
import logging
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from utils import load_config, check_cuda

# Configure logging to use existing app.log
logging.basicConfig(
    filename="app.log",  # Now logs to app.log
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load configuration with error handling
try:
    config = load_config()
    logging.info("Configuration file loaded successfully.")
except FileNotFoundError:
    logging.error("config.json file not found. Ensure the file exists in the root directory.")
    config = None
except json.JSONDecodeError:
    logging.error("Invalid JSON format in config.json. Check for syntax errors.")
    config = None
except Exception as e:
    logging.error(f"Unexpected error while loading config.json: {e}")
    config = None

if config:
    try:
        train_file = config.get("train_file")
        model_checkpoint = config.get("model_checkpoint", "gpt2")
        output_dir = config.get("output_dir", "./gpt2_finetuned")

        if not train_file or not os.path.exists(train_file):
            logging.error(f"Training file not found: {train_file}")
            raise FileNotFoundError(f"Training file not found: {train_file}")

        # Load tokenizer and model
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
            model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
            model.resize_token_embeddings(len(tokenizer))
            tokenizer.pad_token = tokenizer.eos_token
            logging.info("Tokenizer and model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model/tokenizer: {e}")
            raise RuntimeError(f"Error loading model/tokenizer: {e}")

        # Load dataset
        try:
            dataset = load_dataset("json", data_files=train_file, split="train")
            dataset = dataset.train_test_split(test_size=0.1)
            logging.info("Dataset loaded and split successfully.")
        except FileNotFoundError:
            logging.error("Dataset file is missing or incorrect path provided.")
            raise FileNotFoundError("Dataset file is missing or incorrect path provided.")
        except KeyError as e:
            logging.error(f"Missing required dataset keys: {e}")
            raise KeyError(f"Missing required dataset keys: {e}")
        except Exception as e:
            logging.error(f"Error processing dataset: {e}")
            raise RuntimeError(f"Error processing dataset: {e}")

        # Tokenization function
        def tokenize_function(examples):
            combined_texts = [p + " " + c for p, c in zip(examples["prompt"], examples["completion"])]
            return tokenizer(combined_texts, truncation=True, padding=True)

        # Tokenize dataset
        try:
            tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "completion"])
            logging.info("Dataset tokenization completed successfully.")
        except KeyError as e:
            logging.error(f"Dataset missing expected keys: {e}")
            raise KeyError(f"Dataset missing expected keys: {e}")
        except Exception as e:
            logging.error(f"Error tokenizing dataset: {e}")
            raise RuntimeError(f"Error tokenizing dataset: {e}")

        # Data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Check for GPU availability
        device = check_cuda()
        use_fp16 = True if device == "cuda" else False
        logging.info(f"Using device: {device}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=config.get("batch_size", 2),
            per_device_eval_batch_size=config.get("batch_size", 2),
            num_train_epochs=config.get("num_train_epochs", 5),
            learning_rate=config.get("learning_rate", 5e-6),
            warmup_steps=config.get("warmup_steps", 250),
            weight_decay=config.get("weight_decay", 0.1),
            logging_dir=config.get("logging_dir", "./logs"),
            logging_steps=config.get("logging_steps", 10),
            save_total_limit=config.get("save_total_limit", 3),
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            fp16=use_fp16,
            lr_scheduler_type="cosine",
            report_to="none",
        )

        logging.info("Training arguments initialized.")

        # Trainer with Early Stopping
        try:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["test"],
                data_collator=data_collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            )

            logging.info("Training started.")
            trainer.train()
            logging.info("Training completed successfully.")

        except torch.cuda.OutOfMemoryError:
            logging.error("GPU out of memory. Reduce batch size or switch to CPU mode.")
            raise MemoryError("GPU out of memory. Reduce batch size or switch to CPU mode.")
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise RuntimeError(f"Error during training: {e}")

        # Save model and tokenizer
        try:
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info("Model training and saving completed successfully.")
            print("Model training and saving completed successfully.")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise RuntimeError(f"Error saving model: {e}")

    except FileNotFoundError as e:
        logging.error(f"File Error: {e}")
        print(f"File Error: {e}")
    except KeyError as e:
        logging.error(f"Dataset Key Error: {e}")
        print(f"Dataset Key Error: {e}")
    except MemoryError as e:
        logging.error(f"Memory Error: {e}")
        print(f"Memory Error: {e}")
    except RuntimeError as e:
        logging.error(f"Runtime Error: {e}")
        print(f"Runtime Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")
else:
    logging.critical("Training aborted due to configuration loading issues.")
    print("Training aborted due to configuration loading issues.")
