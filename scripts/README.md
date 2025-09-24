# Database Management Scripts

This directory contains scripts to manage PostgreSQL and MongoDB databases for the ARA Demo project.

## Scripts Overview

### 1. `clear_databases.py` - Clear Database Data
Clears all data from both PostgreSQL and MongoDB databases while preserving the table structure.

**Usage:**
```bash
# Interactive mode (asks for confirmation)
python scripts/clear_databases.py

# Skip confirmation prompt
python scripts/clear_databases.py --confirm
```

### 2. `reset_databases.py` - Reset Database Schema
Provides options to either clear data or completely reset the database schema.

**Usage:**
```bash
# Clear data only (default)
python scripts/reset_databases.py --mode clear

# Drop and recreate all tables
python scripts/reset_databases.py --mode reset

# Skip confirmation prompt
python scripts/reset_databases.py --mode reset --confirm
```

### 3. `clear_databases.sh` - Convenient Shell Script
A bash wrapper that provides easy access to the Python scripts.

**Usage:**
```bash
# Clear data but keep schema
./scripts/clear_databases.sh clear

# Drop and recreate all tables
./scripts/clear_databases.sh reset

# Show help
./scripts/clear_databases.sh help
```

## When to Use Each Script

### Use `clear` mode when:
- You want to start fresh with existing schema
- You have the new `pdf_uid` schema already in place
- You want to preserve table structure but remove all data

### Use `reset` mode when:
- You need to completely rebuild the database schema
- You're migrating from the old `pdf_hash` to new `pdf_uid` format
- You want to ensure a completely clean slate

## Database Configuration

The scripts use environment variables for database connection settings. Make sure your `.env` file contains:

```env
# PostgreSQL Configuration
BIORXIV_DB_HOST=localhost
BIORXIV_DB_PORT=9999
BIORXIV_DB_NAME=postgres
BIORXIV_DB_USER=postgres
BIORXIV_DB_PASSWORD=postgres

# MongoDB Configuration
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_USERNAME=admin
MONGODB_PASSWORD=admin123
MONGODB_DATABASE=biorxiv_gemini
MONGODB_COLLECTION=gemini_responses
```

## Safety Features

- **Confirmation Prompts**: All scripts ask for confirmation before making changes
- **Detailed Logging**: All operations are logged with timestamps
- **Error Handling**: Scripts will rollback changes if errors occur
- **Connection Testing**: Scripts verify database connections before proceeding

## Migration from pdf_hash to pdf_uid

If you're migrating from the old `pdf_hash` system to the new `pdf_uid` system:

1. **Backup your data** (if needed)
2. **Run the reset script** to drop old tables:
   ```bash
   ./scripts/clear_databases.sh reset
   ```
3. **Run your new schema initialization** (e.g., Docker Compose or SQL scripts)
4. **Start your application** with the new `pdf_uid` format

## Troubleshooting

### Connection Issues
- Ensure your databases are running
- Check that environment variables are set correctly
- Verify network connectivity to database hosts

### Permission Issues
- Ensure the database user has appropriate permissions
- For PostgreSQL: user needs `TRUNCATE`, `DROP`, and `CREATE` privileges
- For MongoDB: user needs read/write access to the database

### Script Errors
- Check the logs for detailed error messages
- Ensure all required Python packages are installed
- Verify the script is run from the project root directory

## Examples

### Complete Fresh Start
```bash
# Drop everything and start over
./scripts/clear_databases.sh reset

# Then run your schema initialization
docker-compose up -d postgres mongodb
```

### Clear Data Only
```bash
# Keep schema, clear data
./scripts/clear_databases.sh clear
```

### Automated (CI/CD)
```bash
# Skip confirmation for automation
python scripts/reset_databases.py --mode reset --confirm
```

### 4. `complete_migration.py` - Complete Schema Migration
Handles the complete migration from pdf_hash to pdf_uid format, including schema changes, data migration, and cleanup.

**Usage:**
```bash
# Preview complete migration
python scripts/complete_migration.py --dry-run

# Run complete migration
python scripts/complete_migration.py

# Run specific steps only
python scripts/complete_migration.py --steps schema
python scripts/complete_migration.py --steps data
python scripts/complete_migration.py --steps cleanup

# Skip confirmation prompt
python scripts/complete_migration.py --confirm
```

**What it does:**
- **Schema Analysis**: Detects current database state
- **Column Creation**: Adds pdf_uid columns to all tables
- **Data Migration**: Moves data from pdf_hash to pdf_uid format
- **Constraint Updates**: Updates foreign keys and indexes
- **Cleanup**: Removes old pdf_hash columns

