from pyspark.sql import SparkSession
from pyspark.sql.functions import initcap, col, round
from pyspark.sql.types import *
from datetime import datetime

class PysparkProcessor:
    def __init__(self):
        self.spark = SparkSession.builder.appName('Transform').getOrCreate()

    def process_data(self, file_path, output_path):
        # Charger les données depuis le fichier CSV
        df = self.spark.read.csv(file_path, header=True)

        # Définir le schéma des données
        base_fields = [StructField("Asset", StringType(), True)]
        date_fields = [StructField(date, FloatType(), True) for date in df.columns[1:]]
        schema = StructType(base_fields + date_fields)

        # Appliquer le schéma aux données
        df = self.spark.read.csv(file_path, header=True, schema=schema)

        # Effectuer la transformation
        value_col = df.columns[-1]
        yesterday_value_col = df.columns[-2]
        week_ago_col = df.columns[-7]
        month_ago_col = df.columns[-29]

        df = df.withColumn("Day", round(((col(value_col) - col(yesterday_value_col)) / col(yesterday_value_col)) * 100, 2)
                ).withColumn('Value', col(value_col)
                ).withColumn('Week', round(((col(value_col) - col(week_ago_col)) / col(week_ago_col)) * 100, 2)
                ).withColumn('Month', round(((col(value_col) - col(month_ago_col)) / col(month_ago_col)) * 100, 2))

        df = df.select('Asset', 'Value', 'Day', 'Week', 'Month')

        df = df.withColumn("Day", df["Day"].cast(FloatType())
            ).withColumn("Week", df["Week"].cast(FloatType())
            ).withColumn("Month", df["Month"].cast(FloatType())
            ).withColumn("Asset", initcap("Asset"))

        # Sauvegarder les données transformées
        df.write.mode('overwrite').parquet(output_path)
