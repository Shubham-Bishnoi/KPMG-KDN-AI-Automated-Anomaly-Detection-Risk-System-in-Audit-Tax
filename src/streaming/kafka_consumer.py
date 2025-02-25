from confluent_kafka import Consumer, KafkaException, Producer
import json
from src.models.fraud_detector import is_fraudulent

# Kafka configuration
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "transactions"
ALERT_TOPIC = "fraud_alerts"
GROUP_ID = "fraud_detection_group"

consumer = Consumer({
    'bootstrap.servers': KAFKA_BROKER,
    'group.id': GROUP_ID,
    'auto.offset.reset': 'earliest'
})
consumer.subscribe([KAFKA_TOPIC])

producer = Producer({'bootstrap.servers': KAFKA_BROKER})

def process_transaction(transaction):
    """Process transaction & detect fraud"""
    print(f"Processing transaction: {transaction}")
    
    if is_fraudulent(transaction):
        print("🚨 Fraud detected! Sending alert...")
        alert_json = json.dumps({"fraudulent_transaction": transaction})
        producer.produce(ALERT_TOPIC, key=str(transaction["transaction_id"]), value=alert_json)
        producer.flush()

if __name__ == "__main__":
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaException._PARTITION_EOF:
                    continue
                else:
                    print(f"Consumer error: {msg.error()}")
                    break
            transaction_data = json.loads(msg.value().decode('utf-8'))
            process_transaction(transaction_data)
    except KeyboardInterrupt:
        print("Stopping consumer...")
    finally:
        consumer.close()
