-- pgvector nello stesso database dei dati relazionali: gli embedding si
-- filtrano per tenant con la stessa WHERE del resto, stanno nella stessa
-- transazione e nello stesso backup.
CREATE EXTENSION IF NOT EXISTS vector;

-- Ricerca testuale per il recupero ibrido: gli articoli di legge e i nomi
-- propri delle notizie sono i casi in cui la similarita' vettoriale e' debole
-- e il confronto lessicale e' forte.
CREATE EXTENSION IF NOT EXISTS pg_trgm;
