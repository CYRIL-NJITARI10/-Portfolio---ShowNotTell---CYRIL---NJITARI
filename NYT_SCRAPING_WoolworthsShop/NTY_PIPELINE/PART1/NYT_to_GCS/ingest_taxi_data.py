#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import argparse
from sqlalchemy import create_engine
import dask.dataframe as dd
import time

def main(params):
    user = params.user
    password = params.password
    host = params.host
    port = params.port
    db = params.db
    table_name = params.table_name
    url = params.url

    file_extension = os.path.splitext(url)[1]
    if file_extension == ".parquet":
        file_name = "output.parquet"
    elif file_extension == ".csv":
        file_name = "output.csv"
    else:
        raise ValueError("Unsupported file type!")

    os.system(f"wget {url} -O {file_name}")

    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    engine = create_engine(connection_str)

    try:
        if file_extension == ".parquet":
            ddf = dd.read_parquet(file_name, blocksize="10MB")
        elif file_extension == ".csv":
            ddf = dd.from_pandas(pd.read_csv(file_name), npartitions=5)
        
        print(ddf.to_delayed())

        for partition in ddf.to_delayed():
            start_time = time.time()
            
            df = partition.compute()  # Convert Dask partition to pandas DataFrame
            df.to_sql(table_name, engine, if_exists='append', index=False, chunksize=1000)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"Partition inserted in {duration:.2f} seconds.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up resources
        engine.dispose()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest parquet data to postgres')
    parser.add_argument('--user', help='user name for postgres')
    parser.add_argument('--password', help='password for postgres')
    parser.add_argument('--host',  help='host for postgres')
    parser.add_argument('--port',  help='port for postgres')
    parser.add_argument('--db', help='database name for postgres') 
    parser.add_argument('--table-name', help='name of the table where we will write the results to ')
    parser.add_argument('--url', help='url of the parquet file')

    args = parser.parse_args()
    main(args)
