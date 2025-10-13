# Curl Commands for Testing Data Summary

Complete guide for testing the data summarization feature.

## Prerequisites

1. **Start the API server:**
   ```bash
   cd /home/piotr/projects/ara_demo/ara_demo_api
   uvicorn app.main:app --reload
   ```

2. **Ensure test data exists:**
   ```bash
   ls test_data.csv  # Should exist
   ```

3. **Set your session ID:**
   ```bash
   export SESSION_ID="test-session-123"
   ```

---

## Step-by-Step Testing

### 1. Health Check

```bash
curl http://localhost:8000/health | jq '.'
```

**Expected Output:**
```json
{
  "success": true,
  "message": "System is healthy",
  "data": {
    "status": "healthy",
    "services": {
      "database": "connected"
    }
  }
}
```

---

### 2. Create Session

```bash
curl -X POST "http://localhost:8000/sessions" \
  -H "X-Session-ID: $SESSION_ID" | jq '.'
```

**Expected Output:**
```json
{
  "message": "Session created",
  "session_id": "test-session-123",
  "created_at": "2025-10-10T10:00:00"
}
```

---

### 3. Upload CSV File

```bash
curl -X POST "http://localhost:8000/files" \
  -H "X-Session-ID: $SESSION_ID" \
  -F "file=@test_data.csv" | jq '.'
```

**Expected Output:**
```json
{
  "file_id": "894d41e0-1285-459d-86ca-5443da3943dd",
  "file_name": "test_data.csv",
  "file_size": 350,
  "file_type": "text/csv",
  "upload_timestamp": "2025-10-10T10:00:00",
  "metadata": {
    "encoding": "utf-8",
    "delimiter": ",",
    "rows": 8,
    "columns": 5
  }
}
```

**Save the file_id:**
```bash
export FILE_ID="894d41e0-1285-459d-86ca-5443da3943dd"  # Use actual ID from response
```

---

### 4. Get Data Summary WITH LLM Descriptions (Main Test)

```bash
curl "http://localhost:8000/files/$FILE_ID/summary?infer_descriptions=true" | jq '.'
```

**Expected Output:**
```json
{
  "row_count": 8,
  "column_count": 5,
  "columns": [
    {
      "name": "species_name",
      "type": "object",
      "non_null_count": 8,
      "null_count": 0,
      "null_percentage": 0.0,
      "unique_values": 8,
      "description": "Name of the observed species or organism",
      "value_counts": {
        "Oak Tree": 1,
        "Pine Tree": 1,
        ...
      }
    },
    {
      "name": "observation_count",
      "type": "int64",
      "non_null_count": 8,
      "null_count": 0,
      "null_percentage": 0.0,
      "unique_values": 8,
      "description": "Number of individual observations recorded for each species",
      "statistics": {
        "mean": 78.75,
        "median": 71.0,
        "std": 42.15,
        "min": 23.0,
        "max": 156.0,
        "q25": 45.75,
        "q75": 103.5
      }
    },
    ...
  ],
  "column_descriptions": {
    "species_name": "Name of the observed species or organism",
    "observation_count": "Number of individual observations recorded for each species",
    "habitat_type": "Type of habitat where the species was observed",
    "temperature_celsius": "Average temperature in degrees Celsius",
    "rainfall_mm": "Annual rainfall in millimeters"
  },
  "file_metadata": {
    "file_id": "894d41e0-1285-459d-86ca-5443da3943dd",
    "filename": "test_data.csv",
    "file_size": 350,
    "encoding": "utf-8",
    "delimiter": ","
  }
}
```

---

### 5. Get Data Summary WITHOUT LLM Descriptions (Faster)

```bash
curl "http://localhost:8000/files/$FILE_ID/summary?infer_descriptions=false" | jq '.'
```

**This skips LLM calls** - faster, but no descriptions.

---

### 6. Get Specific Column Descriptions Only

```bash
curl "http://localhost:8000/files/$FILE_ID/summary?infer_descriptions=true" | \
  jq '.column_descriptions'
```

**Expected Output:**
```json
{
  "species_name": "Name of the observed species or organism",
  "observation_count": "Number of individual observations recorded for each species",
  "habitat_type": "Type of habitat where the species was observed",
  "temperature_celsius": "Average temperature in degrees Celsius",
  "rainfall_mm": "Annual rainfall in millimeters"
}
```

---

### 7. Get Column Statistics Only

```bash
curl "http://localhost:8000/files/$FILE_ID/summary" | \
  jq '.columns[] | select(.statistics) | {name, statistics}'
```

**Shows only numeric columns with their statistics.**

---

### 8. Check Row and Column Counts

```bash
curl "http://localhost:8000/files/$FILE_ID/summary" | \
  jq '{rows: .row_count, columns: .column_count}'
```

**Expected Output:**
```json
{
  "rows": 8,
  "columns": 5
}
```

---

### 9. Get Sample Data

```bash
curl "http://localhost:8000/files/$FILE_ID/summary" | \
  jq '.sample_data[:3]'
```

**Shows first 3 rows of data.**

---

### 10. Get File Metadata

```bash
curl "http://localhost:8000/files/$FILE_ID" | jq '.'
```

**Expected Output:**
```json
{
  "file_id": "894d41e0-1285-459d-86ca-5443da3943dd",
  "filename": "test_data.csv",
  "file_size": 350,
  "file_type": "text/csv",
  "created_at": "2025-10-10T10:00:00",
  "extra_metadata": {
    "encoding": "utf-8",
    "delimiter": ",",
    "rows": 8,
    "columns": 5,
    "column_names": ["species_name", "observation_count", ...]
  }
}
```

---

## Complete Testing Workflow

```bash
# Set variables
export SESSION_ID="my-test-session"
export API_URL="http://localhost:8000"

# 1. Create session
curl -X POST "$API_URL/sessions" -H "X-Session-ID: $SESSION_ID"

# 2. Upload file and capture file_id
UPLOAD_RESP=$(curl -s -X POST "$API_URL/files" \
  -H "X-Session-ID: $SESSION_ID" \
  -F "file=@test_data.csv")

export FILE_ID=$(echo $UPLOAD_RESP | jq -r '.file_id')
echo "File ID: $FILE_ID"

# 3. Get summary with LLM descriptions
curl "$API_URL/files/$FILE_ID/summary?infer_descriptions=true" | jq '.'

# 4. Extract just the descriptions
curl "$API_URL/files/$FILE_ID/summary" | jq '.column_descriptions'
```

---

## Using the Test Script

**Run the automated test:**
```bash
cd /home/piotr/projects/ara_demo/ara_demo_api
./test_api.sh
```

**The script will:**
- ✅ Test API health
- ✅ Create a session
- ✅ Upload test_data.csv
- ✅ Get summary with LLM descriptions
- ✅ Get summary without LLM descriptions
- ✅ Display results

---

## Testing Different Scenarios

### Test with Different CSV Files

```bash
# Create another test file
cat > scientific_data.csv << 'EOF'
experiment_id,treatment,response_var,p_value,effect_size
E001,Control,12.5,0.045,0.32
E002,Treatment_A,15.8,0.012,0.58
E003,Treatment_B,18.2,0.003,0.75
EOF

# Upload it
curl -X POST "http://localhost:8000/files" \
  -H "X-Session-ID: $SESSION_ID" \
  -F "file=@scientific_data.csv" | jq '.file_id'

# Get summary
export FILE_ID2=$(echo $UPLOAD_RESP | jq -r '.file_id')
curl "http://localhost:8000/files/$FILE_ID2/summary" | jq '.column_descriptions'
```

### Test with Large File

```bash
# Generate large CSV
python3 -c "
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'id': range(1000),
    'value': np.random.randn(1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'timestamp': pd.date_range('2024-01-01', periods=1000)
})
df.to_csv('large_data.csv', index=False)
"

# Upload
curl -X POST "http://localhost:8000/files" \
  -H "X-Session-ID: $SESSION_ID" \
  -F "file=@large_data.csv"
```

---

## Troubleshooting

### Connection Refused
```bash
# Check if server is running
curl http://localhost:8000/health

# If not, start it
uvicorn app.main:app --reload
```

### Session Not Found
```bash
# Create session first
curl -X POST "http://localhost:8000/sessions" \
  -H "X-Session-ID: your-session-id"
```

### File Not Found
```bash
# Check if file exists
ls test_data.csv

# Or create it
cat > test_data.csv << 'EOF'
col1,col2,col3
1,2,3
4,5,6
EOF
```

### jq Not Installed
```bash
# Install jq for pretty JSON
sudo apt-get install jq

# Or view raw JSON
curl http://localhost:8000/health
```

---

## Performance Comparison

**Without LLM descriptions (fast):**
```bash
time curl -s "http://localhost:8000/files/$FILE_ID/summary?infer_descriptions=false" > /dev/null
# Expected: < 1 second
```

**With LLM descriptions (slower, but richer):**
```bash
time curl -s "http://localhost:8000/files/$FILE_ID/summary?infer_descriptions=true" > /dev/null
# Expected: 2-5 seconds (depends on OpenAI API latency)
```

---

## Example: Extract Specific Information

```bash
# Get only column names
curl -s "$API_URL/files/$FILE_ID/summary" | jq -r '.columns[].name'

# Get only numeric columns
curl -s "$API_URL/files/$FILE_ID/summary" | jq '.columns[] | select(.statistics)'

# Get columns with missing data
curl -s "$API_URL/files/$FILE_ID/summary" | jq '.columns[] | select(.null_count > 0)'

# Get row and column counts
curl -s "$API_URL/files/$FILE_ID/summary" | jq '{rows: .row_count, cols: .column_count}'

# Get all descriptions as a simple list
curl -s "$API_URL/files/$FILE_ID/summary" | \
  jq -r '.column_descriptions | to_entries[] | "\(.key): \(.value)"'
```

---

## Integration Testing

**Test the complete workflow:**

```bash
#!/bin/bash
SESSION_ID="integration-test-$(date +%s)"

# 1. Upload
FILE_RESP=$(curl -s -X POST "http://localhost:8000/files" \
  -H "X-Session-ID: $SESSION_ID" \
  -F "file=@test_data.csv")

FILE_ID=$(echo $FILE_RESP | jq -r '.file_id')

# 2. Get summary
SUMMARY=$(curl -s "http://localhost:8000/files/$FILE_ID/summary")

# 3. Verify all expected fields exist
echo "Checking summary structure..."
echo $SUMMARY | jq -e '.row_count' && echo "✓ row_count exists"
echo $SUMMARY | jq -e '.columns' && echo "✓ columns exists"
echo $SUMMARY | jq -e '.column_descriptions' && echo "✓ descriptions exists"
echo $SUMMARY | jq -e '.sample_data' && echo "✓ sample_data exists"

echo "✓ All fields present!"
```

---

## Quick Reference

```bash
# Health check
curl http://localhost:8000/health

# Create session
curl -X POST http://localhost:8000/sessions -H "X-Session-ID: test-123"

# Upload file
curl -X POST http://localhost:8000/files -H "X-Session-ID: test-123" -F "file=@data.csv"

# Get summary (with descriptions)
curl "http://localhost:8000/files/{file_id}/summary?infer_descriptions=true"

# Get summary (without descriptions - faster)
curl "http://localhost:8000/files/{file_id}/summary?infer_descriptions=false"

# Get file info
curl http://localhost:8000/files/{file_id}

# Delete file
curl -X DELETE http://localhost:8000/files/{file_id}
```

Replace `{file_id}` with actual file ID from upload response.

