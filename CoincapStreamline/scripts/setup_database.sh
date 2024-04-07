#!/bin/bash

# Paramètres de connexion à la base de données
DB_USER="Coincap_Cyril"
DB_PASS="Coincap_Cyril"
DB_HOST="localhost" # ou l'adresse IP de votre serveur de base de données
DB_PORT="5432"

# Nom du fichier SQL contenant les commandes de création
SQL_FILE="init_db.sql"

# Exécute les commandes SQL depuis le fichier
PGPASSWORD=$DB_PASS psql -U $DB_USER -h $DB_HOST -p $DB_PORT -f $SQL_FILE


