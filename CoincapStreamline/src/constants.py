# Configuration générale et constantes

URL_API = "https://api.coincap.io/v2/assets"
MAX_LIMIT = 100
KAFKA_SERVER = 'kafka:9092'
KAFKA_TOPIC = 'coincap_topic'
POSTGRES_TABLE = 'crypto_data'
#POSTGRES_URL = "jdbc:postgresql://localhost:5432/Coincap_db"
POSTGRES_URL = "jdbc:postgresql://coincap-postgres:5432/Coincap_db"
POSTGRES_PROPERTIES = {
    "user": "Coincap_Cyril",
    "password": "Coincap_Cyril",
    "driver": "org.postgresql.Driver",
    "dbname": "Coincap_db",
    "host": "coincap-postgres"
}

CHECKPOINT_LOCATION = "/home/jovyan/checkpoints"