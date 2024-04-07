from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, to_timestamp, month, year, count, hour
from pyspark.sql.types import TimestampType
from datetime import datetime
#from google.cloud import storage
from pyspark.sql.window import Window
from pyspark.sql.functions import rank

def create_success_file(bucket_name):
    print(f"Création du fichier _SUCCESS dans le bucket {bucket_name}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('_SUCCESS_END')
    blob.upload_from_string('')
    print("Fichier _SUCCESS_END créé.")

def add_years_handling_leap(date):
    new_year = date.year + 3
    if date.month == 2 and date.day == 29:
        if (new_year % 4 != 0) or (new_year % 100 == 0 and new_year % 400 != 0):
            return datetime(new_year, date.month, 28, date.hour, date.minute, date.second)
        else:
            return datetime(new_year, date.month, date.day, date.hour, date.minute, date.second)
    else:
        return datetime(new_year, date.month, date.day, date.hour, date.minute, date.second)

spark = SparkSession.builder.appName("DateAdjustmentExample").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

#la fonction UDF
add_years_udf = udf(add_years_handling_leap, TimestampType())
spark.udf.register("add_years_udf", add_years_handling_leap)

bucket_name = 'crimes_bucket_final'
out_bucket = 'cyrilo_bucket_final'

#base_path = f"gs://{out_bucket}/"
#gcs_path = f'gs://{bucket_name}/Crimes_-_2001_to_Present.csv'

df = spark.read.csv("Crimes_-_2001_to_Present.csv", header=True, inferSchema=True)

# Renomme les colonnes avec des caractères invalides
df = df.withColumnRenamed("Location Description", "Location_Description") \
       .withColumnRenamed("Primary Type", "Primary_Type") \
       .withColumnRenamed("Arrest", "Arrest_Flag")

# Appliquer les transformations
df = df.withColumn("Date", to_timestamp(col("Date"), "MM/dd/yyyy hh:mm:ss a"))
df = df.withColumn("Date", add_years_udf(col("Date")))

df = df.withColumn("Year", col("Year") + 3)

#df.select("Date", "Year").show(5, truncate=False)

current_year = datetime.now().year

# Question 1
df_filtered = df.filter((year(col("Date")) >= (current_year - 5)) & (year(col("Date")) <= current_year))
result_q1 = df_filtered.groupBy(month(col("Date")).alias("Month")) \
                       .agg(count("*").alias("TotalCrimes")) \
                       .orderBy("Month")
result_q1.show()
#result_q1.write.parquet(base_path + "question1_output.parquet")

# Question 2
theft_df = df.filter((col("Primary_Type") == "THEFT") & (year(col("Date")) >= current_year - 2) & (year(col("Date")) <= current_year ))


windowSpec = Window.partitionBy("Year").orderBy(col("TotalThefts").desc())

theft_ranked_df = theft_df.groupBy("Year", "Location_Description") \
                          .agg(count("*").alias("TotalThefts")) \
                          .withColumn("Rank", rank().over(windowSpec))

# Filtrer pour garder seulement le top 10 pour chaque année
result_q2 = theft_ranked_df.filter(col("Rank") <= 10)

# Afficher le résultat
result_q2.show(30)  

  # Question 3
result_q3 = df.groupBy(year(col("Date")).alias("Year")) \
              .agg(count("*").alias("TotalCrimes")) \
              .orderBy("Year")
result_q3.show()
#result_q3.write.parquet(base_path + "question3_output.parquet")

#Question 4
result_q4 = df.filter((hour(col("Date")) >= 22) | (hour(col("Date")) <= 4)) \
              .groupBy("Location_Description") \
              .agg(count("*").alias("TotalCrimes")) \
              .orderBy(col("TotalCrimes").desc())
result_q4.show()
#result_q4.write.parquet(base_path + "question4_output.parquet")

         # Question 5
result_q5 = df.filter((year(col("Date")) >= 2016) & (year(col("Date")) <= 2019) & (col("Arrest_Flag") == True)) \
              .groupBy("Primary_Type") \
              .agg(count("*").alias("TotalArrests")) \
              .orderBy(col("TotalArrests").desc())
result_q5.show()
#result_q5.write.parquet(base_path + "question5_output.parquet")

dataframes = [result_q1, result_q2, result_q3, result_q4, result_q5]

for df in dataframes:
    print(df.count())

#create_success_file(out_bucket)

spark.stop()
