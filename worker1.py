import torch
import json
import redis
import pika
import logging
import time
from utils import load_config, load_model_tokenizer, check_cuda

# Configure logging
logging.basicConfig(
    filename="worker.log", 
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load configuration
try:
    config = load_config()
    logging.info("Configuration file loaded successfully.")
except Exception as e:
    logging.critical(f"Failed to load configuration: {e}")
    exit(1)  # Exit if config cannot be loaded

# Load model and tokenizer
try:
    model, tokenizer = load_model_tokenizer(config.get("model_path", ""))
    device = check_cuda()
    model.to(device)
    logging.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logging.critical(f"Failed to load model/tokenizer: {e}")
    exit(1)  # Exit if model loading fails

# Initialize Redis connection with retry logic
def connect_redis(retries=5, delay=5):
    for attempt in range(retries):
        try:
            client = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=5)
            client.ping()  # Check if Redis is responsive
            logging.info("Connected to Redis successfully.")
            return client
        except redis.ConnectionError as e:
            logging.error(f"Failed to connect to Redis (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)  # Wait before retrying
    logging.critical("Could not connect to Redis after multiple attempts. Exiting.")
    exit(1)

redis_client = connect_redis()

# Initialize RabbitMQ connection with retry logic
def connect_rabbitmq(retries=5, delay=5):
    for attempt in range(retries):
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host='localhost', heartbeat=30)
            )
            channel = connection.channel()
            channel.queue_declare(queue="text_generation")
            logging.info("Connected to RabbitMQ successfully.")
            return connection, channel
        except pika.exceptions.AMQPConnectionError as e:
            logging.error(f"Failed to connect to RabbitMQ (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)  # Wait before retrying
    logging.critical("Could not connect to RabbitMQ after multiple attempts. Exiting.")
    exit(1)

rabbitmq_connection, channel = connect_rabbitmq()

def process_request(ch, method, properties, body):
    """Processes messages from RabbitMQ queue."""
    try:
        task_data = json.loads(body)
        task_id = task_data.get("task_id")
        prompt = task_data.get("prompt", "").strip()

        if not task_id or not prompt:
            logging.warning(f"Invalid request: {task_data}")
            redis_client.set(task_id, json.dumps({"status": "Failed", "error": "Invalid request format"}))
            return

        logging.info(f"Processing task: {task_id}")

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_length=config.get("max_length", 50),
                temperature=config.get("temperature", 1.0),
                top_k=config.get("top_k", 50),
                top_p=config.get("top_p", 0.9),
                repetition_penalty=config.get("repetition_penalty", 1.2),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Store result in Redis
            redis_client.set(task_id, json.dumps({"status": "Completed", "generated_text": generated_text}))
            logging.info(f"Task {task_id} completed successfully.")

        except torch.cuda.OutOfMemoryError:
            logging.error("GPU memory exhausted. Switching to CPU and retrying.")
            device = torch.device("cpu")  # Switch to CPU
            model.to(device)
            redis_client.set(task_id, json.dumps({"status": "Failed", "error": "GPU Out of Memory"}))

        except Exception as e:
            logging.error(f"Error processing task {task_id}: {e}")
            redis_client.set(task_id, json.dumps({"status": "Failed", "error": str(e)}))

    except json.JSONDecodeError:
        logging.error("Failed to decode JSON request.")
    except Exception as e:
        logging.critical(f"Unexpected error in process_request: {e}")

# Graceful restart logic for RabbitMQ
def start_consumer():
    while True:
        try:
            channel.basic_consume(queue="text_generation", on_message_callback=process_request, auto_ack=True)
            logging.info("Worker started. Listening for messages...")
            channel.start_consuming()
        except pika.exceptions.ConnectionClosedByBroker:
            logging.warning("RabbitMQ connection closed. Reconnecting...")
            global rabbitmq_connection, channel
            rabbitmq_connection, channel = connect_rabbitmq()
        except pika.exceptions.StreamLostError:
            logging.warning("RabbitMQ connection lost. Reconnecting...")
            rabbitmq_connection, channel = connect_rabbitmq()
        except Exception as e:
            logging.critical(f"Unexpected error in RabbitMQ consumer: {e}")
            time.sleep(5)  # Prevent crash loops

start_consumer()
