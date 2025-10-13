#!/bin/bash

# Test script for ARA Demo API
# Tests file upload and data summarization

set -e  # Exit on error

API_URL="http://localhost:8000"
SESSION_ID="test-session-$(date +%s)"
TEST_FILE="test_data.csv"

echo "=========================================="
echo "ARA Demo API - Test Script"
echo "=========================================="
echo ""
echo "API URL: $API_URL"
echo "Session ID: $SESSION_ID"
echo "Test File: $TEST_FILE"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print step
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Function to print error
print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Test 1: Health Check
print_step "Step 1: Testing API health..."
HEALTH_RESPONSE=$(curl -s "$API_URL/health")
echo "$HEALTH_RESPONSE" | jq '.' 2>/dev/null || echo "$HEALTH_RESPONSE"

if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    print_success "API is healthy"
else
    print_error "API health check failed"
    exit 1
fi
echo ""

# Test 2: Create Session
print_step "Step 2: Creating session..."
SESSION_RESPONSE=$(curl -s -X POST "$API_URL/sessions" \
    -H "X-Session-ID: $SESSION_ID")

echo "$SESSION_RESPONSE" | jq '.' 2>/dev/null || echo "$SESSION_RESPONSE"

if echo "$SESSION_RESPONSE" | grep -q "Session created\|Session already exists"; then
    print_success "Session created/verified"
else
    print_error "Session creation failed"
    exit 1
fi
echo ""

# Test 3: Upload CSV File
print_step "Step 3: Uploading CSV file..."
UPLOAD_RESPONSE=$(curl -s -X POST "$API_URL/files" \
    -H "X-Session-ID: $SESSION_ID" \
    -F "file=@$TEST_FILE")

echo "$UPLOAD_RESPONSE" | jq '.' 2>/dev/null || echo "$UPLOAD_RESPONSE"

# Extract file_id from response
FILE_ID=$(echo "$UPLOAD_RESPONSE" | jq -r '.file_id' 2>/dev/null)

if [ -n "$FILE_ID" ] && [ "$FILE_ID" != "null" ]; then
    print_success "File uploaded successfully"
    echo "   File ID: $FILE_ID"
else
    print_error "File upload failed"
    echo "$UPLOAD_RESPONSE"
    exit 1
fi
echo ""

# Test 4: Get File Summary (with LLM descriptions)
print_step "Step 4: Getting data summary with LLM-inferred descriptions..."
SUMMARY_RESPONSE=$(curl -s "$API_URL/files/$FILE_ID/summary?infer_descriptions=true")

echo "$SUMMARY_RESPONSE" | jq '.' 2>/dev/null || echo "$SUMMARY_RESPONSE"

if echo "$SUMMARY_RESPONSE" | grep -q "column_descriptions\|columns"; then
    print_success "Summary generated successfully"
    
    # Display column descriptions if available
    echo ""
    echo "Column Descriptions:"
    echo "$SUMMARY_RESPONSE" | jq -r '.column_descriptions // empty | to_entries[] | "  - \(.key): \(.value)"' 2>/dev/null || echo "  (jq not available to parse)"
else
    print_error "Summary generation failed"
fi
echo ""

# Test 5: Get File Summary (without LLM descriptions - faster)
print_step "Step 5: Getting data summary without LLM descriptions..."
SUMMARY_BASIC=$(curl -s "$API_URL/files/$FILE_ID/summary?infer_descriptions=false")

echo "$SUMMARY_BASIC" | jq '.columns[] | {name, type, unique_values, null_count}' 2>/dev/null || echo "$SUMMARY_BASIC"

if echo "$SUMMARY_BASIC" | grep -q "columns"; then
    print_success "Basic summary generated successfully"
else
    print_error "Basic summary failed"
fi
echo ""

# Test 6: Get File Info
print_step "Step 6: Getting file metadata..."
FILE_INFO=$(curl -s "$API_URL/files/$FILE_ID")

echo "$FILE_INFO" | jq '.' 2>/dev/null || echo "$FILE_INFO"

if echo "$FILE_INFO" | grep -q "$FILE_ID"; then
    print_success "File metadata retrieved"
else
    print_error "Failed to get file metadata"
fi
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Session ID: $SESSION_ID"
echo "File ID: $FILE_ID"
echo ""
echo "Next steps to test:"
echo "  1. Enrich summary with hypothesis:"
echo "     curl -X POST '$API_URL/files/$FILE_ID/enrich' \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"hypothesis\": \"Your hypothesis here\"}'"
echo ""
echo "  2. View in browser:"
echo "     Open: $API_URL/docs"
echo "     Try endpoint: GET /files/$FILE_ID/summary"
echo ""
print_success "All tests completed!"

