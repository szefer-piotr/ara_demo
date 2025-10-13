# Quick Testing Guide

## 🚀 Quick Start

### 1. Start the API

```bash
cd /home/piotr/projects/ara_demo/ara_demo_api
uvicorn app.main:app --reload
```

### 2. Run Automated Tests

**Option A: Bash Script** (requires jq)
```bash
./test_api.sh
```

**Option B: Python Script** (more portable)
```bash
python3 test_summary.py
```

**Option C: Python Script without LLM** (faster)
```bash
python3 test_summary.py --no-llm
```

---

## 📋 Manual Testing with curl

### Quick Test (3 commands)

```bash
# 1. Create session
curl -X POST http://localhost:8000/sessions \
  -H "X-Session-ID: quick-test"

# 2. Upload file
UPLOAD=$(curl -X POST http://localhost:8000/files \
  -H "X-Session-ID: quick-test" \
  -F "file=@test_data.csv")

# 3. Get file_id and fetch summary
FILE_ID=$(echo $UPLOAD | jq -r '.file_id')
curl "http://localhost:8000/files/$FILE_ID/summary" | jq '.'
```

### With Pretty Output

```bash
# Set variables
SESSION="test-$(date +%s)"
API="http://localhost:8000"

# Create session
echo "Creating session..."
curl -s -X POST "$API/sessions" -H "X-Session-ID: $SESSION" | jq '.'

# Upload
echo -e "\nUploading file..."
RESP=$(curl -s -X POST "$API/files" \
  -H "X-Session-ID: $SESSION" \
  -F "file=@test_data.csv")

FILE_ID=$(echo $RESP | jq -r '.file_id')
echo "File ID: $FILE_ID"

# Get summary
echo -e "\nGetting summary..."
curl -s "$API/files/$FILE_ID/summary" | jq '.column_descriptions'
```

---

## 📊 Testing Different Features

### Test WITHOUT LLM Descriptions (Fast)

```bash
curl "http://localhost:8000/files/$FILE_ID/summary?infer_descriptions=false" | jq '.'
```

⏱️ **Time:** < 1 second  
💰 **Cost:** Free (no LLM calls)  
📦 **Output:** Statistics only, no descriptions

### Test WITH LLM Descriptions (Rich)

```bash
curl "http://localhost:8000/files/$FILE_ID/summary?infer_descriptions=true" | jq '.'
```

⏱️ **Time:** 2-5 seconds  
💰 **Cost:** ~$0.001 per request  
📦 **Output:** Statistics + AI-generated descriptions

---

## 🔍 Inspecting Results

### Get Only Column Descriptions

```bash
curl -s "http://localhost:8000/files/$FILE_ID/summary" | \
  jq '.column_descriptions'
```

### Get Only Column Names and Types

```bash
curl -s "http://localhost:8000/files/$FILE_ID/summary" | \
  jq '.columns[] | {name, type, description}'
```

### Get Statistics for Numeric Columns

```bash
curl -s "http://localhost:8000/files/$FILE_ID/summary" | \
  jq '.columns[] | select(.statistics) | {name, statistics}'
```

### Get Sample Data

```bash
curl -s "http://localhost:8000/files/$FILE_ID/summary" | \
  jq '.sample_data[:3]'  # First 3 rows
```

---

## 🧪 Testing with Different Data

### Create Test CSV on the Fly

```bash
cat > ecology_data.csv << 'EOF'
site_id,species_richness,temperature,precipitation,elevation
S001,24,18.5,850,120
S002,31,16.2,1200,450
S003,19,22.1,600,80
S004,28,17.8,950,220
EOF

# Upload and test
curl -X POST http://localhost:8000/files \
  -H "X-Session-ID: ecology-test" \
  -F "file=@ecology_data.csv"
```

### Test with Missing Data

```bash
cat > missing_data.csv << 'EOF'
id,value,category
1,10.5,A
2,,B
3,15.2,
4,12.1,C
EOF

curl -X POST http://localhost:8000/files \
  -H "X-Session-ID: missing-test" \
  -F "file=@missing_data.csv"
```

---

## 🐛 Troubleshooting

### API Not Running

```bash
# Check if running
curl http://localhost:8000/health

# If not, start it
cd ara_demo_api
uvicorn app.main:app --reload
```

### Session Error

```
{"detail":"Session not found"}
```

**Solution:** Create session first:
```bash
curl -X POST http://localhost:8000/sessions -H "X-Session-ID: your-id"
```

### File Not Found

```bash
# Create test data
echo "col1,col2,col3
1,2,3
4,5,6" > test_data.csv

# Verify it exists
cat test_data.csv
```

### jq Not Installed

```bash
# Install jq
sudo apt-get install jq

# Or use Python
curl http://localhost:8000/health | python3 -m json.tool
```

---

## 📈 Expected Response Structure

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
      "description": "AI-generated description here",
      "value_counts": {...}
    },
    {
      "name": "observation_count",
      "type": "int64",
      "statistics": {
        "mean": 78.75,
        "median": 71.0,
        "std": 42.15,
        "min": 23.0,
        "max": 156.0,
        "q25": 45.75,
        "q75": 103.5
      },
      "description": "AI-generated description here"
    }
  ],
  "column_descriptions": {
    "species_name": "Name of the observed species",
    "observation_count": "Number of observations recorded",
    ...
  },
  "dtypes": {...},
  "missing_values": {...},
  "sample_data": [...],
  "file_metadata": {
    "file_id": "uuid",
    "filename": "test_data.csv",
    "encoding": "utf-8",
    "delimiter": ","
  },
  "created_at": "2025-10-10T10:00:00"
}
```

---

## 🎯 One-Liner Tests

```bash
# Complete workflow in one line
SESSION="test-$(date +%s)" && \
curl -sX POST http://localhost:8000/sessions -H "X-Session-ID: $SESSION" && \
FILE_ID=$(curl -sX POST http://localhost:8000/files -H "X-Session-ID: $SESSION" -F "file=@test_data.csv" | jq -r '.file_id') && \
curl -s "http://localhost:8000/files/$FILE_ID/summary" | jq '.column_descriptions'
```

---

## 🔗 Next Steps

After successful summary testing:

1. **Test hypothesis enrichment:**
   ```bash
   curl -X POST "$API/files/$FILE_ID/enrich" \
     -H "Content-Type: application/json" \
     -d '{"hypothesis": "Temperature affects species diversity"}'
   ```

2. **Generate analysis plan:**
   ```bash
   curl -X POST "$API/generate-plan" \
     -H "Content-Type: application/json" \
     -d '{"file_id": "'$FILE_ID'", "hypothesis": "..."}'
   ```

3. **Execute analysis:**
   ```bash
   curl -X POST "$API/execute-step" \
     -H "Content-Type: application/json" \
     -d '{"step_id": "1", "file_id": "'$FILE_ID'"}'
   ```

Happy testing! 🎉

