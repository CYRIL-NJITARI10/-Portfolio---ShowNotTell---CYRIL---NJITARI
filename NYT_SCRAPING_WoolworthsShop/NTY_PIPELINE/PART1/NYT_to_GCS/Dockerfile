FROM python:3.11

RUN apt-get update && apt-get install -y wget
RUN pip install pandas sqlalchemy dask[complete] fastparquet pyarrow psycopg2-binary 

WORKDIR /app

COPY ingest_taxi_data.py ingest_taxi_data.py

RUN chmod +x ingest_taxi_data.py

ENTRYPOINT [ "python", "ingest_taxi_data.py" ]
