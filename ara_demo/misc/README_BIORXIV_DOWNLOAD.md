# BioRxiv TDM Download Script

This script downloads BioRxiv papers from the last month using the official Text and Data Mining (TDM) S3 bucket and uploads them to your MinIO bucket. **NEW**: Now includes automatic unpacking of `.meca` files to extract individual PDF, XML, and supplementary files.

## 🚀 Features

- **Official Access**: Uses the official BioRxiv TDM S3 bucket (`s3://biorxiv-src-monthly`)
- **Automatic Month Detection**: Automatically detects and downloads from the last month
- **Metadata Extraction**: Extracts title, authors, DOI, and category from `.meca` files
- **MinIO Integration**: Uploads papers and metadata to your MinIO bucket
- **Requester-Pays**: Handles AWS requester-pays bucket properly
- **Error Handling**: Robust error handling with detailed logging
- **🆕 Automatic Unpacking**: Extracts individual files from `.meca` packages
- **🆕 File Organization**: Organizes unpacked files with proper content types

## 📋 Prerequisites

1. **AWS Account**: You need an AWS account with S3 access
2. **AWS Credentials**: Access key and secret key for S3 operations
3. **MinIO**: Running MinIO instance (local or remote)
4. **Python Dependencies**: Install required packages

## 🛠️ Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up AWS Credentials

You have several options for providing AWS credentials:

#### Option A: Environment Variables
```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
```

#### Option B: .env File (Recommended)
```bash
# Copy the example config
cp biorxiv_config.env.example .env

# Edit .env with your credentials
nano .env
```

#### Option C: AWS CLI Configuration
```bash
aws configure
```

### 3. Verify MinIO is Running

The script defaults to `localhost:9002` for MinIO. Make sure your MinIO instance is running:

```bash
# Check if MinIO is accessible
curl http://localhost:9002/minio/health/live
```

## 🚀 Usage

### Basic Usage

Download papers from the last month to a MinIO bucket:

```bash
python download_biorxiv_env.py
```

### Download and Unpack (Recommended)

Download papers and automatically unpack `.meca` files to extract individual components:

```bash
python download_biorxiv_env.py --unpack-meca
```

### Unpack Only (No Download)

If you already have `.meca` files and just want to unpack them:

```bash
python download_biorxiv_env.py --unpack-only --month-folder July_2025
```

### Advanced Usage

```bash
# Download with custom settings and unpack
python download_biorxiv_env.py \
    --minio-bucket my-papers \
    --unpack-meca \
    --max-unpack 25

# Use custom MinIO endpoint
python download_biorxiv_env.py \
    --minio-endpoint my-minio:9000 \
    --minio-bucket papers \
    --unpack-meca

# Use custom AWS credentials
python download_biorxiv_env.py \
    --aws-access-key YOUR_KEY \
    --aws-secret-key YOUR_SECRET \
    --minio-bucket papers \
    --unpack-meca
```

## 📁 File Structure

### Original Structure (Download Only)
```
biorxiv-papers/
├── July_2025/
│   ├── Current_Content_July_2025_paper1.meca
│   ├── Current_Content_July_2025_paper1.metadata.json
│   ├── Current_Content_July_2025_paper2.meca
│   ├── Current_Content_July_2025_paper2.metadata.json
│   └── ...
```

### Enhanced Structure (With Unpacking)
```
biorxiv-papers/
├── July_2025/
│   ├── Current_Content_July_2025_paper1.meca
│   ├── Current_Content_July_2025_paper1.metadata.json
│   └── ...
├── unpacked/
│   ├── July_2025_paper1/
│   │   ├── manifest.xml
│   │   ├── content_paper.pdf
│   │   ├── content_paper.xml
│   │   ├── content_supplementary.txt
│   │   ├── content_figure1.png
│   │   └── metadata.json
│   ├── July_2025_paper2/
│   │   ├── manifest.xml
│   │   ├── content_paper.pdf
│   │   ├── content_paper.xml
│   │   └── metadata.json
│   └── ...
```

## 📊 What Gets Downloaded and Unpacked

### .meca Files (Original)
- These are actually ZIP files containing:
  - `manifest.xml` - Paper metadata
  - `content/` folder with:
    - PDF file
    - XML file
    - Supplementary files (if any)
    - Images (if any)

### Unpacked Files (New)
- **Individual PDF files**: Direct access to research papers
- **XML files**: Structured content for text mining
- **Supplementary materials**: Additional data and figures
- **Enhanced metadata**: JSON with unpacked file information
- **Proper content types**: Files tagged with correct MIME types

### Metadata Files
- JSON files with extracted information:
  - Title, Authors, DOI, Category
  - Abstract (if available)
  - List of all files in the package
  - **🆕 Unpacked file locations**
  - **🆕 Content type information**

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | AWS access key | Required |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | Required |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `MINIO_ENDPOINT` | MinIO endpoint | `localhost:9002` |
| `MINIO_ACCESS_KEY` | MinIO access key | `piotrminio` |
| `MINIO_SECRET_KEY` | MinIO secret key | `piotrminio` |
| `MAX_FILES` | Maximum files to download | `100` |
| `MINIO_BUCKET` | MinIO bucket name | `biorxiv-papers` |

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--minio-bucket` | MinIO bucket name | Required |
| `--unpack-meca` | Download and unpack .meca files | False |
| `--unpack-only` | Only unpack existing files (skip download) | False |
| `--month-folder` | Month folder to unpack (with --unpack-only) | Required if --unpack-only |
| `--max-unpack` | Maximum .meca files to unpack | No limit |
| `--max-files` | Maximum files to download | `100` |
| `--aws-access-key` | AWS access key | Environment var |
| `--aws-secret-key` | AWS secret key | Environment var |
| `--minio-endpoint` | MinIO endpoint | `localhost:9002` |
| `--minio-access-key` | MinIO access key | `piotrminio` |
| `--minio-secret-key` | MinIO secret key | `piotrminio` |
| `--verbose` | Enable verbose logging | False |

## 🔍 How It Works

### Download Process
1. **Month Detection**: Automatically determines the last month (e.g., "July_2025")
2. **S3 Listing**: Lists all `.meca` files in `Current_Content/{month}/`
3. **Download**: Downloads each file from BioRxiv S3 (requester-pays)
4. **Metadata Extraction**: Extracts metadata from the `.meca` file
5. **MinIO Upload**: Uploads both the `.meca` file and metadata JSON

### Unpacking Process (New)
1. **File Download**: Downloads `.meca` file from MinIO to temporary location
2. **ZIP Extraction**: Extracts all files from the `.meca` package
3. **Content Analysis**: Determines proper content types for each file
4. **Individual Upload**: Uploads each file individually to MinIO
5. **Enhanced Metadata**: Creates comprehensive metadata with file locations
6. **Cleanup**: Removes temporary files

## 💰 Cost Considerations

- **BioRxiv S3**: Uses requester-pays bucket (you pay AWS data transfer costs)
- **Data Transfer**: Typically minimal cost for reasonable usage
- **Storage**: MinIO storage costs depend on your setup
- **🆕 Unpacking**: No additional AWS costs, only MinIO storage for extracted files

## 🚨 Important Notes

### Requester-Pays Bucket
- The BioRxiv S3 bucket is a **requester-pays** bucket
- You will be charged for data transfer by AWS
- Costs are typically minimal for reasonable usage

### Rate Limiting
- Be respectful of AWS and BioRxiv resources
- The script includes built-in delays and error handling
- Consider using `--max-files` to limit downloads

### File Formats
- `.meca` files are actually ZIP files
- Each contains PDF, XML, and supplementary materials
- Metadata is extracted from `manifest.xml`
- **🆕 Unpacked files maintain original structure and content types**

### Unpacking Benefits
- **Direct Access**: Individual PDF and XML files for easy processing
- **Content Types**: Proper MIME type tagging for web applications
- **Text Mining**: XML files ready for automated analysis
- **Storage Efficiency**: Choose which components to keep

## 🐛 Troubleshooting

### Common Issues

#### AWS Credentials Error
```
❌ Error: AWS credentials are required
```
**Solution**: Set your AWS credentials in environment variables or `.env` file

#### MinIO Connection Error
```
❌ Failed to create/check MinIO bucket
```
**Solution**: Check if MinIO is running and accessible at the specified endpoint

#### S3 Access Error
```
❌ Failed to list files in Current_Content/...
```
**Solution**: Verify AWS credentials and S3 permissions

#### No Files Found
```
⚠️ No .meca files found in July_2025
```
**Solution**: The month folder might not exist yet (BioRxiv updates monthly)

#### Unpacking Errors
```
❌ Failed to unpack Current_Content_July_2025_paper1.meca
```
**Solution**: Check if the .meca file is corrupted or incomplete

### Debug Mode

Enable verbose logging to see detailed information:

```bash
python download_biorxiv_env.py --verbose --minio-bucket papers --unpack-meca
```

## 📚 Examples

### Example 1: First-Time Setup with Unpacking
```bash
# 1. Copy configuration
cp biorxiv_config.env.example .env

# 2. Edit configuration
nano .env

# 3. Download and unpack
python download_biorxiv_env.py --unpack-meca
```

### Example 2: Download to Custom Bucket and Unpack
```bash
python download_biorxiv_env.py --minio-bucket my-research-papers --unpack-meca
```

### Example 3: Limited Download and Unpacking
```bash
python download_biorxiv_env.py --max-files 25 --max-unpack 10 --unpack-meca
```

### Example 4: Unpack Existing Files Only
```bash
python download_biorxiv_env.py --unpack-only --month-folder July_2025
```

### Example 5: Unpack with File Limit
```bash
python download_biorxiv_env.py --unpack-only --month-folder July_2025 --max-unpack 5
```

## 🔗 Related Files

- `download_biorxiv_last_month.py` - Main download script
- `download_biorxiv_env.py` - Environment wrapper with unpacking support
- `biorxiv_config.env.example` - Configuration template
- `requirements.txt` - Python dependencies

## 📄 License

This script is provided as-is for educational and research purposes. Please respect BioRxiv's terms of service and AWS usage policies.

## 🤝 Contributing

Feel free to submit issues or pull requests to improve the script. Consider:

- Adding support for specific date ranges
- Implementing parallel downloads
- Adding more metadata extraction options
- Supporting other storage backends
- **🆕 Adding support for other archive formats**
- **🆕 Implementing selective unpacking (PDF only, XML only, etc.)**
