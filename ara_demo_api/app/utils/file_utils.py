import pandas as pd
import chardet
from typing import Tuple
import logging
from io import BytesIO
import csv

from app.config import settings

logger = logging.getLogger(__name__)


def robust_read_csv(file_content: bytes, filename: str) -> Tuple[pd.DataFrame, str, str]:
    """
    Read CSV from bytes, handling multiple encodings - Synchronous.

    Parameters:
    - file_content: Raw file bytes
    - filename: Original filename

    Returns: (DataFrame, encoding_used, delimiter_used)
    """
    detected = chardet.detect(file_content).get('encoding')
    logger.debug(f"Chardet guess for {filename}: {detected}")

    encodings = [
        detected,
        "utf-8-sig",
        "utf-8",
        "cp1250",
        "iso-8859-2",
        "latin1",
    ]

    # Filter out None values
    encodings = [enc for enc in encodings if enc]

    for enc in encodings:
        try:
            # Try to decode a sample to check encoding validity
            sample = file_content[:4096].decode(enc, errors="strict")
            
            # Try to detect delimiter
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t")
                delim = dialect.delimiter
            except Exception:
                delim = ","
            
            # Try to read the full CSV
            df = pd.read_csv(
                BytesIO(file_content),
                delimiter=delim,
                encoding=enc,
                engine="python"
            )
            logger.info(f"Successfully read {filename} with encoding={enc}, delimiter='{delim}'")
            return df, enc, delim
            
        except Exception as e:
            logger.debug(f"Failed to read {filename} with encoding={enc}: {e}")
    
    # If all encodings failed
    raise UnicodeDecodeError(
        "robust_read_csv",
        b"",
        0,
        1,
        f"Unable to decode {filename} with any known encoding."
    )
    


def validate_csv_file(filename: str) -> bool:
    """Check if file has .csv extension"""
    return filename.lower().endswith('.csv')


def csv_to_bytes(df: pd.DataFrame, encoding: str = 'utf-8') -> bytes:
    """Convert DataFrame to CSV bytes"""
