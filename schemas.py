# schemas.py

from pydantic import BaseModel
from typing import List

class ColumnSummary(BaseModel):
    column_name: str
    description: str
    type: str
    unique_value_count: int


class DatasetSummary(BaseModel):
    columns: List[ColumnSummary]