FROM jupyter/pyspark-notebook:latest

# Installer psycopg2-binary
RUN pip install psycopg2-binary

# Définir la variable d'environnement PYTHONPATH
ENV PYTHONPATH="/home/jovyan:$PYTHONPATH"

USER root
RUN mkdir -p /home/jovyan/src
COPY ./src /home/jovyan/src
COPY ./data /home/jovyan/data
COPY ./local-checkpoints /home/jovyan/checkpoints
USER jovyan