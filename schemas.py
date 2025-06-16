# schemas.py

from pydantic import BaseModel
from typing import List

class ColumnSummary(BaseModel):
    column_name: str
    description: str
    type: str
    unique_value_count: int

    def __str__(self) -> str:
            return (f"{self.column_name} ({self.type}): "
                    f"{self.description} – unique={self.unique_value_count}")

class DatasetSummary(BaseModel):
    columns: List[ColumnSummary]

    def __str__(self) -> str:
        return " | ".join(map(str, self.columns))


class AnalysisStep(BaseModel):
    step_title: str
    step_text: str

class AnalysisPlan(BaseModel):
    steps: List[AnalysisStep]