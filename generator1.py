import torch
import logging
import json
import uuid
import redis
import pika
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import load_config

# Configure logging
logging.basicConfig(
    filename="app.log", 
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize FastAPI
app = FastAPI()

# Load configuration with error handling
try:
    config = load_config()
    if not config or "host" not in config or "port" not in config:
        raise ValueError("Invalid configuration file.")
    logging.info("Configuration file loaded successfully.")
except (FileNotFoundError, ValueError) as e:
    logging.critical(f"Failed to load config: {e}")
    config = None  # Prevent FastAPI from starting if config is invalid

# Initialize Redis connection with retry logic
def connect_redis(retries=5, delay=5):
    for attempt in range(retries):
        try:
            client = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=5)
            client.ping()  # Test connection
            logging.info("Connected to Redis successfully.")
            return client
        except redis.ConnectionError as e:
            logging.error(f"Failed to connect to Redis (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)
    logging.critical("Could not connect to Redis after multiple attempts. Exiting.")
    exit(1)

redis_client = connect_redis()

# Initialize RabbitMQ connection with retry logic
def connect_rabbitmq(retries=5, delay=5):
    for attempt in range(retries):
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', heartbeat=30))
            channel = connection.channel()
            channel.queue_declare(queue="text_generation")
            logging.info("Connected to RabbitMQ successfully.")
            return connection, channel
        except pika.exceptions.AMQPConnectionError as e:
            logging.error(f"Failed to connect to RabbitMQ (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)
    logging.critical("Could not connect to RabbitMQ after multiple attempts. Exiting.")
    exit(1)

rabbitmq_connection, channel = connect_rabbitmq()

# Define request model
class TextRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(request: TextRequest):
    """API Endpoint to enqueue a text generation task"""
    global rabbitmq_connection, channel  # Declare global variables at the start

    try:
        if not request.prompt.strip():
            logging.warning("Invalid request: Prompt is empty.")
            raise HTTPException(status_code=400, detail="Prompt must be a non-empty string.")

        task_id = str(uuid.uuid4())  # Generate unique task ID
        task_data = json.dumps({"task_id": task_id, "prompt": request.prompt.strip()})

        # Store initial status in Redis with retry
        for _ in range(3):  # Retry up to 3 times
            try:
                redis_client.set(task_id, json.dumps({"status": "Pending"}))
                break
            except redis.ConnectionError as e:
                logging.warning(f"Redis set failed: {e}. Retrying...")
                time.sleep(2)
        else:
            logging.error("Failed to store task in Redis after retries.")
            raise HTTPException(status_code=500, detail="Internal server error.")

        # Publish task to RabbitMQ
        try:
            channel.basic_publish(exchange='', routing_key='text_generation', body=task_data)
        except pika.exceptions.AMQPConnectionError as e:
            logging.error(f"RabbitMQ publish failed: {e}. Reconnecting...")

            # Reconnect RabbitMQ
            rabbitmq_connection, channel = connect_rabbitmq()

            # Retry publishing the task after reconnection
            channel.basic_publish(exchange='', routing_key='text_generation', body=task_data)

        logging.info(f"Task {task_id} enqueued successfully.")

        return {
            "task_id": task_id,
            "message": "Processing started. Use /task_status/{task_id} to check result."
        }

    except HTTPException as e:
        raise e  # Re-raise FastAPI exceptions
    except Exception as e:
        logging.error(f"Unexpected error in /generate: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/task_status/{task_id}")
def get_task_status(task_id: str):
    """API Endpoint to fetch the task result from Redis"""
    try:
        result = redis_client.get(task_id)

        if not result:
            raise HTTPException(status_code=404, detail="Invalid task ID or expired result.")

        result = json.loads(result)

        if result["status"] == "Pending":
            return {"task_id": task_id, "status": "Pending", "message": "Processing..."}

        return {"task_id": task_id, "status": "Completed", "generated_text": result["generated_text"]}

    except redis.ConnectionError as e:
        logging.error(f"Redis connection error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
    except json.JSONDecodeError:
        logging.error("Failed to decode Redis response.")
        raise HTTPException(status_code=500, detail="Internal server error.")
    except Exception as e:
        logging.error(f"Unexpected error in /task_status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

if __name__ == "__main__":
    import uvicorn
    if config:
        logging.info(f"Starting FastAPI server on {config['host']}:{config['port']}")
        uvicorn.run(app, host=config["host"], port=config["port"])
    else:
        logging.critical("Server could not start due to missing or invalid configuration.")
