from pyspark.sql import SparkSession
from pyspark.sql.functions import col 

bucket_name = 'crimes_bucket_final_a2'
temporary_gcs_bucket = 'crimes_bucket_final_a2'
bigquery_table = 'cyril-njitari:archi2_dataset_final.raw_crime_data'

spark = SparkSession.builder.appName('Data Transfer to BigQuery').getOrCreate()

df = spark.read.option("header", "true").csv(f'gs://{bucket_name}/Crimes_-_2001_to_Present.csv')

df_cleaned = df.select([col("`" + c + "`").alias(c.replace(' ', '_').replace('.', '').replace('/', '_')) for c in df.columns])

df_cleaned.write.format('bigquery') \
    .option('table', bigquery_table) \
    .option('temporaryGcsBucket', temporary_gcs_bucket) \
    .mode('overwrite') \
    .save()

