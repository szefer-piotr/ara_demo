# Data Migration Process Example

## Before Migration (Current State)
```sql
-- Table: extracted_texts
CREATE TABLE extracted_texts (
    id SERIAL PRIMARY KEY,
    pdf_hash VARCHAR(64) UNIQUE NOT NULL,  -- OLD COLUMN
    s3_key VARCHAR(255) NOT NULL,
    pdf_name VARCHAR(255) NOT NULL,
    full_text TEXT NOT NULL,
    -- ... other columns
);

-- Sample data:
-- id | pdf_hash                    | s3_key    | pdf_name
-- 1  | a1b2c3d4e5f6...            | biorxiv_1 | paper1.pdf
-- 2  | f6e5d4c3b2a1...            | biorxiv_2 | supplement1.pdf
```

## Step 1: Add New Column
```sql
-- Add pdf_uid column alongside pdf_hash
ALTER TABLE extracted_texts ADD COLUMN pdf_uid VARCHAR(500);

-- Table now looks like:
-- id | pdf_hash                    | pdf_uid | s3_key    | pdf_name
-- 1  | a1b2c3d4e5f6...            | NULL    | biorxiv_1 | paper1.pdf
-- 2  | f6e5d4c3b2a1...            | NULL    | biorxiv_2 | supplement1.pdf
```

## Step 2: Migrate Data
```sql
-- Copy data from pdf_hash to pdf_uid with new format
UPDATE extracted_texts 
SET pdf_uid = s3_key || '_' || pdf_name || 
    CASE WHEN pdf_name LIKE '%supplement%' THEN '_supplement' ELSE '' END
WHERE pdf_hash IS NOT NULL;

-- Table now looks like:
-- id | pdf_hash                    | pdf_uid                    | s3_key    | pdf_name
-- 1  | a1b2c3d4e5f6...            | biorxiv_1_paper1.pdf       | biorxiv_1 | paper1.pdf
-- 2  | f6e5d4c3b2a1...            | biorxiv_2_supplement1.pdf_supplement | biorxiv_2 | supplement1.pdf
```

## Step 3: Update Constraints
```sql
-- Add new unique constraint on pdf_uid
ALTER TABLE extracted_texts ADD CONSTRAINT extracted_texts_pdf_uid_unique UNIQUE (pdf_uid);

-- Update foreign key references in other tables
-- (This happens for all related tables)
```

## Step 4: Drop Old Column
```sql
-- Remove the old pdf_hash column
ALTER TABLE extracted_texts DROP COLUMN pdf_hash;

-- Final table:
-- id | pdf_uid                    | s3_key    | pdf_name
-- 1  | biorxiv_1_paper1.pdf       | biorxiv_1 | paper1.pdf
-- 2  | biorxiv_2_supplement1.pdf_supplement | biorxiv_2 | supplement1.pdf
```

## Key Points:
- ✅ **Same database** - no separate databases needed
- ✅ **Same table** - columns are added/removed in place
- ✅ **Data preserved** - all existing data is migrated
- ✅ **Atomic operation** - can be rolled back if needed
- ✅ **Zero downtime** - happens within existing database
