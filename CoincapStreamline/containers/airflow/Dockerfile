FROM apache/airflow:2.8.2

# Assurez-vous que le fichier requirements.txt est dans le contexte de build Docker, à la racine.
COPY requirements.txt /requirements.txt

# Installez les dépendances Python à partir de requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
