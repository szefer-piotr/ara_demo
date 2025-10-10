import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import chardet
from typing import Tuple
import logging
from io import BytesIO
import csv

from app.config import settings

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)


def _read_csv_sync(file_content: bytes, filename: str) -> Tuple[pd.DataFrame, str, str]:
    """Internal synchronous CSV reader - runs in thread pool"""
    detected = chardet.detect(file_content).get('encoding')
    logger.debug(f"Chardet guess: {detected}")

    encodings = [
        detected,
        "utf-8-sig",
        "utf-8",
        "cp1250",
        "iso-8859-2",
        "latin1",
    ]

    encodings = [enc for enc in encodings if enc]

    for enc in encodings:
        try:
            sample = file_content[:4096].decode(enc, errors="strict")
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t")
                delim = dialect.delimiter
            except Exception:
                delim = ","
            df = pd.read_csv(BytesIO(file_content), delimiter=delim, encoding=enc, engine="python")
            logging.debug(f"Successfully read with encoding = {enc}, delimiter='{delim}'")
            return df, enc, delim
        except Exception as e:
            logging.debug(f"Failed with encoding={enc}: {e}")
        
    raise UnicodeDecodeError(
        "robust_read_csv", b"", 0, 1,
        "Unable to decode with any known encoding."
    )


async def robust_read_csv(file_content: bytes, filename: str) -> Tuple[pd.DataFrame, str, str]:
    """
    Async CSV reader that offloads blocking operations to a thread pool.

    Parameters:
    - file_content: Raw file bytes
    - filename: Original filename

    Returns: (DataFrame, encoding_used, delimiter_used)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _read_csv_sync, file_content, filename)
    


def validate_csv_file(filename: str) -> bool:
    """Check if file has .csv extension"""
    return filename.lower().endswith('.csv')


def csv_to_bytes(df: pd.DataFrame, encoding: str = 'utf-8') -> bytes:
    """Convert DataFrame to CSV bytes"""
