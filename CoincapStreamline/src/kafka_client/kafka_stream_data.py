from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
import requests
import json
import time
from src.constants import URL_API, KAFKA_SERVER, KAFKA_TOPIC

class CoinCapProducer:
    def __init__(self, server=KAFKA_SERVER, topic=KAFKA_TOPIC, num_partitions=4):
        self.server = server
        self.num_partitions = num_partitions
        self.topic = topic
        self.producer = KafkaProducer(bootstrap_servers=[server], value_serializer=lambda v: json.dumps(v).encode('utf-8'))
        self.create_topic()

    def create_topic(self):
        admin_client = KafkaAdminClient(bootstrap_servers=self.server, client_id='producer_client')
        # Récupérer la liste des topics existants
        existing_topics = admin_client.list_topics()
        if self.topic not in existing_topics:
            topic_list = [NewTopic(name=self.topic, num_partitions=self.num_partitions, replication_factor=1)]
            try:
                admin_client.create_topics(new_topics=topic_list, validate_only=False)
                print(f"Topic {self.topic} created")
            except Exception as e:
                print(f"Could not create topic {self.topic}: {e}")
        else:
            print(f"Topic {self.topic} already exists")

    def fetch_data(self):
        response = requests.get(URL_API)
        if response.status_code == 200:
            return response.json()['data']
        else:
            print("Failed to retrieve data")
            return []

    def publish_data(self, data):
        for item in data:
            self.producer.send(self.topic, item)
            self.producer.flush()
            print(f"Sent asset data: {item['id']}")

if __name__ == "__main__":
    producer = CoinCapProducer()
    while True:
        data = producer.fetch_data()
        producer.publish_data(data)
        time.sleep(600)  # Wait for 10 minutes before repeating
