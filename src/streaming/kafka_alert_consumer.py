from confluent_kafka import Consumer, KafkaException
import json

# Kafka configuration
KAFKA_BROKER = "localhost:9092"
ALERT_TOPIC = "fraud_alerts"
GROUP_ID = "fraud_alert_group"

consumer = Consumer({
    'bootstrap.servers': KAFKA_BROKER,
    'group.id': GROUP_ID,
    'auto.offset.reset': 'earliest'
})
consumer.subscribe([ALERT_TOPIC])

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
            alert_data = json.loads(msg.value().decode('utf-8'))
            print(f"🚨 FRAUD ALERT RECEIVED: {alert_data}")
    except KeyboardInterrupt:
        print("Stopping alert consumer...")
    finally:
        consumer.close()
