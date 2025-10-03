# BioRxiv Manual Search Script

A Python script that builds BioRxiv search URLs manually by mimicking their internal search structure. This approach works around the limitations of their official API by generating search URLs that you can manually visit in your browser.

## Features

- **Manual URL Construction**: Builds search URLs by mimicking BioRxiv's internal search structure
- **Flexible Search Options**: Search by terms, subjects, date ranges, and more
- **Browser Integration**: Optionally opens search URLs directly in your default web browser
- **CLI Interface**: Full command-line interface with argument parsing
- **Advanced Encoding Options**: Configurable URL encoding to match different share link styles
- **No External Dependencies**: Uses only Python standard library

## How It Works

Instead of using a non-existent API endpoint, this script:
1. **Constructs search URLs** that match BioRxiv's internal search structure
2. **Generates properly encoded URLs** with all search parameters
3. **Optionally opens URLs** in your default web browser
4. **Provides URL-only output** for scripting and automation

**Note**: Due to Cloudflare protection on BioRxiv's website, automated scraping is not possible. This script generates URLs that you manually visit to view results.

## Installation

**No installation required!** This script uses only Python standard library modules.

Simply download the script and run it:
```bash
python3 search_biorxiv.py --help
```

## Usage

### Command Line Interface

**Basic search (shows URL):**
```bash
python search_biorxiv.py "population ecology"
```

**Search with specific subjects:**
```bash
python search_biorxiv.py "machine learning" -s "Bioinformatics" "Computational Biology"
```

**Search with date range:**
```bash
python search_biorxiv.py "CRISPR" -s "Genetics" -d 2024-01-01 2024-12-31
```

**Open search directly in browser:**
```bash
python search_biorxiv.py "cancer immunotherapy" -s "Cancer Biology" --open
```

**Get URL-only output (for scripting):**
```bash
python search_biorxiv.py "neural networks" -s "Neuroscience" --url-only
```

**Custom number of results:**
```bash
python search_biorxiv.py "neural networks" -s "Neuroscience" -n 20
```

### Command Line Options

- `terms`: Search terms or phrases (required)
- `-s, --subjects`: Subject categories (default: Evolutionary Biology, Plant Biology, Systems Biology, Zoology)
- `-d, --dates`: Date range in YYYY-MM-DD format (default: 2024-01-01 to 2024-12-31)
- `-n, --num_results`: Number of results to return (default: 10)
- `--sort`: Sort order - relevance-rank, date-asc, date-desc (default: relevance-rank)
- `--jcode`: Journal code - biorxiv or medrxiv (default: biorxiv)
- `--match`: Match flag - match-any or match-all (default: match-any)
- `--open, -o`: Open the search URL in your default web browser
- `--url-only`: Output only the URL (useful for scripting)
- `--no-plus`: Disable plus encoding in terms (use spaces instead)
- `--double-encode-commas`: Double-encode commas in terms (mimics some share links)

### Programmatic Usage

```python
from search_biorxiv import BiorxivSearchBuilder, BiorxivSearchOptions

# Create search options
opts = BiorxivSearchOptions(
    terms="population ecology",
    subjects=["Evolutionary Biology", "Ecology"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    numresults=10
)

# Build search URL
builder = BiorxivSearchBuilder()
url = builder.build_search_url(opts)
print(f"Search URL: {url}")

# Optionally open in browser
builder.open_search_in_browser(opts)
```

## URL Construction Details

The script constructs search URLs that include:
- Search terms (with configurable encoding)
- Subject categories
- Date ranges
- Result limits and sorting
- Match flags and journal codes

Example generated URL structure:
```
https://www.biorxiv.org/search/population+ecology+title:population+ecology+title_flags:match-any+...
```

## Examples

### Search for population ecology papers:
```bash
python search_biorxiv.py "population ecology" -s "Evolutionary Biology" "Ecology" -n 5
```

### Search for machine learning in bioinformatics and open in browser:
```bash
python search_biorxiv.py "machine learning" -s "Bioinformatics" "Computational Biology" -n 10 --open
```

### Search for recent CRISPR papers:
```bash
python search_biorxiv.py "CRISPR" -s "Genetics" -d 2024-01-01 2024-12-31 -n 15
```

### Get URL for scripting:
```bash
python search_biorxiv.py "neural networks" -s "Neuroscience" --url-only > search_url.txt
```

## Advanced Features

### URL Encoding Options

- **Plus Encoding**: Replace spaces with `+` in search terms
- **Double-encoded Commas**: Use `%252C` instead of `%2C` for commas (mimics some share links)
- **Safe Encoding**: Properly encode special characters for URL construction

### Subject Categories

Common subject categories include:
- Evolutionary Biology
- Plant Biology
- Systems Biology
- Zoology
- Bioinformatics
- Computational Biology
- Genetics
- Neuroscience
- Cancer Biology
- Microbiology

## Why This Approach?

BioRxiv doesn't provide a public search API, and their website is protected by Cloudflare, so this script:

1. **Reverses their search URL structure** from analyzing their website
2. **Mimics their internal search parameters** to get proper results
3. **Generates valid search URLs** that you can manually visit
4. **Provides a clean CLI interface** for easy URL generation
5. **Integrates with your browser** for seamless searching

## Use Cases

- **Research**: Quickly generate search URLs for specific research topics
- **Automation**: Use `--url-only` to generate URLs for scripts or batch processing
- **Teaching**: Demonstrate how BioRxiv's search system works
- **Documentation**: Create shareable search links for colleagues

## Requirements

- Python 3.6+
- No external dependencies required

## Testing

Test the URL building functionality:
```bash
python search_biorxiv.py "test query" --url-only
```

## Limitations

- **No automated scraping**: Due to Cloudflare protection, results must be viewed manually
- **Browser dependency**: The `--open` option requires a web browser
- **URL length**: Very complex searches may generate very long URLs

## License

This script is provided as-is for educational and research purposes.
