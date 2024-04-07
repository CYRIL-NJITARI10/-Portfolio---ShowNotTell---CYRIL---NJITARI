from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from src.constants import POSTGRES_URL, POSTGRES_PROPERTIES, POSTGRES_TABLE, KAFKA_TOPIC, KAFKA_SERVER, CHECKPOINT_LOCATION
from utils.transformers import get_schema, deduplicate_staging_table, upsert_from_staging_to_main, truncate_staging_table

class CoinCapSparkConsumer:
    def __init__(self, appName="CoinCap Data Streaming", kafkaServer=KAFKA_SERVER, topic=KAFKA_TOPIC, postgresTable=POSTGRES_TABLE):
        self.spark = (SparkSession.builder.appName(appName)
                      .config("spark.executor.instances", "4")
                      .config("spark.executor.cores", "4")
                      .config("spark.executor.memory", "4g")
                      .config("spark.jars", "/home/jovyan/data/postgresql-42.2.5.jar")
                      .getOrCreate())
        self.spark.sparkContext.setLogLevel("ERROR")
        self.kafkaServer = kafkaServer
        self.topic = topic
        self.postgresTable = postgresTable
        self.schema = get_schema()

    def read_from_kafka(self, max_records=100):
        df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafkaServer) \
            .option("subscribe", self.topic) \
            .option("maxOffsetsPerTrigger", max_records) \
            .load()
        return df

    def process_stream(self, df):
        json_df = df.selectExpr("CAST(value AS STRING) as json") \
                    .select(from_json(col("json"), self.schema).alias("data")) \
                    .select("data.*")

        def write_micro_batch_to_postgres(batch_df, epoch_id):
            conn = psycopg2.connect(dbname=POSTGRES_PROPERTIES["dbname"], user=POSTGRES_PROPERTIES["user"], password=POSTGRES_PROPERTIES["password"], host=POSTGRES_PROPERTIES["host"])
            cur = conn.cursor()

            # Écriture du micro-batch dans la table de staging
            (batch_df.write
             .format("jdbc")
             .option("url", POSTGRES_URL)
             .option("dbtable", "crypto_data_staging")
             .option("user", POSTGRES_PROPERTIES["user"])
             .option("password", POSTGRES_PROPERTIES["password"])
             .option("driver", POSTGRES_PROPERTIES["driver"])
             .mode("append")
             .save())

            deduplicate_staging_table(cur)
            upsert_from_staging_to_main(cur, self.postgresTable)
            truncate_staging_table(cur)

            conn.commit()
            cur.close()
            conn.close()

        query = json_df.writeStream \
            .foreachBatch(write_micro_batch_to_postgres) \
            .option("checkpointLocation", CHECKPOINT_LOCATION) \
            .start()
        query.awaitTermination()

if __name__ == "__main__":
    consumer = CoinCapSparkConsumer()
    kafka_df = consumer.read_from_kafka()
    consumer.process_stream(kafka_df)
