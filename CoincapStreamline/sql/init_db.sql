\c Coincap_db;

CREATE TABLE crypto_data (
    id TEXT PRIMARY KEY,
    rank INTEGER,
    symbol TEXT,
    name TEXT,
    supply DECIMAL,
    maxSupply DECIMAL,
    marketCapUsd DECIMAL,
    volumeUsd24Hr DECIMAL,
    priceUsd DECIMAL,
    changePercent24Hr DECIMAL,
    vwap24Hr DECIMAL,
    explorer TEXT
);

CREATE TABLE crypto_data_staging (
    id TEXT,
    rank INTEGER,
    symbol TEXT,
    name TEXT,
    supply DECIMAL,
    maxSupply DECIMAL,
    marketCapUsd DECIMAL,
    volumeUsd24Hr DECIMAL,
    priceUsd DECIMAL,
    changePercent24Hr DECIMAL,
    vwap24Hr DECIMAL,
    explorer TEXT
);
