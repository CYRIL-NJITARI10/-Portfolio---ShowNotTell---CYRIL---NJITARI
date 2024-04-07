# utils/transformers.py

from pyspark.sql.types import StructType, StructField, StringType, DecimalType, IntegerType

# Définition du schéma pour les données JSON
def get_schema():
    return StructType([
        StructField("id", StringType()),
        StructField("rank", IntegerType()),
        StructField("symbol", StringType()),
        StructField("name", StringType()),
        StructField("supply", DecimalType(38, 0)),
        StructField("maxSupply", DecimalType(38, 0)),
        StructField("marketCapUsd", DecimalType(38, 0)),
        StructField("volumeUsd24Hr", DecimalType(38, 0)),
        StructField("priceUsd", DecimalType(38, 0)),
        StructField("changePercent24Hr", DecimalType(38, 2)),
        StructField("vwap24Hr", DecimalType(38, 2)),
        StructField("explorer", StringType())
    ])

import psycopg2

def deduplicate_staging_table(cur):
    deduplicate_sql = """
        DELETE FROM crypto_data_staging
        WHERE ctid NOT IN (
            SELECT min(ctid)
            FROM crypto_data_staging
            GROUP BY id
        );
    """
    cur.execute(deduplicate_sql)

def upsert_from_staging_to_main(cur, postgres_table):
    upsert_sql = f"""
        INSERT INTO {postgres_table} (id, rank, symbol, name, supply, maxSupply, marketCapUsd, volumeUsd24Hr, priceUsd, changePercent24Hr, vwap24Hr, explorer)
        SELECT id, rank, symbol, name, supply, maxSupply, marketCapUsd, volumeUsd24Hr, priceUsd, changePercent24Hr, vwap24Hr, explorer
        FROM crypto_data_staging
        ON CONFLICT (id) DO UPDATE SET
            rank = EXCLUDED.rank,
            symbol = EXCLUDED.symbol,
            name = EXCLUDED.name,
            supply = EXCLUDED.supply,
            maxSupply = EXCLUDED.maxSupply,
            marketCapUsd = EXCLUDED.marketCapUsd,
            volumeUsd24Hr = EXCLUDED.volumeUsd24Hr,
            priceUsd = EXCLUDED.priceUsd,
            changePercent24Hr = EXCLUDED.changePercent24Hr,
            vwap24Hr = EXCLUDED.vwap24Hr,
            explorer = EXCLUDED.explorer;
    """
    cur.execute(upsert_sql)

def truncate_staging_table(cur):
    truncate_sql = "TRUNCATE TABLE crypto_data_staging;"
    cur.execute(truncate_sql)
