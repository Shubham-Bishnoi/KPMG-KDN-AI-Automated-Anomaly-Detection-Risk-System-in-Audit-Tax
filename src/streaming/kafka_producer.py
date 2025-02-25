from confluent_kafka import Producer
import json
import time
import random

# Kafka configuration
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "transactions"

producer = Producer({'bootstrap.servers': KAFKA_BROKER})

def send_transaction():
    transaction = {
        "transaction_id": random.randint(100000, 999999),
        "client_id": random.randint(1000, 5000),
        "amount": round(random.uniform(1.0, 1000.0), 2),
        "merchant_id": random.randint(1000, 5000),
        "timestamp": time.time()
    }
    
    transaction_json = json.dumps(transaction)
    producer.produce(KAFKA_TOPIC, key=str(transaction["transaction_id"]), value=transaction_json)
    producer.flush()
    print(f"Sent transaction: {transaction}")

if __name__ == "__main__":
    while True:
        send_transaction()
        time.sleep(2)
