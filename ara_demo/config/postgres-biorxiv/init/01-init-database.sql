-- Create the biorxiv_processing database and user
-- This script runs automatically when the container starts for the first time

-- Create the user (if it doesn't exist)
-- DO $$
-- BEGIN
--     IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'biorxiv_reader') THEN
--         CREATE ROLE biorxiv_reader WITH LOGIN PASSWORD 'biorxiv_reader_password';
--     END IF;
-- END
-- $$;

-- -- Grant privileges
-- GRANT ALL PRIVILEGES ON DATABASE biorxiv_processing TO biorxiv_reader;
-- GRANT ALL PRIVILEGES ON SCHEMA public TO biorxiv_reader;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO biorxiv_reader;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO biorxiv_reader;

-- -- Set default privileges for future objects
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO biorxiv_reader;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO biorxiv_reader;