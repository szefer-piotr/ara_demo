from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

class ColumnSummary(BaseModel):
    column_name: str = Field(description="The name of the column")
    description: str = Field(description="The description of the column")
    type: str = Field(description="The type of the column")
    unique_value_count: int = Field(description="The number of unique values in the column")

    def __str__(self):
        return f"ColumnSummary(column_name={self.column_name}, description={self.description}, type={self.type}, unique_value_count={self.unique_value_count})"

class DatasetSummary(BaseModel):
    columns: List[ColumnSummary] = Field(description="The columns in the dataset")

    def __str__(self) -> str:
        return " | ".join([str(column) for column in self.columns])


class AnalysisStep(BaseModel):
    step_title: str = Field(description="The title of the step")
    step_text: str = Field(description="The text of the step")


class AnalysisPlan(BaseModel):
    steps: List[AnalysisStep] = Field(description="The steps in the analysis plan")


class HypothesisStatus(str, Enum):
    """Status of the hypothesis"""
    DRAFT = "draft"
    ACCEPTED = "accepted"
    TESTING = "testing"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class HypothesisSchema(BaseModel):
    """Schema for hypothesis data structure"""
    id: Optional[str] = None
    title: str = Field(..., min_length=1, max_length=200, description="Hypothesis title")
    description: str = Field(..., min_length=1, description="Detailed hypothesis description")
    prediction: str = Field(..., min_length=1, description="What you expect to find")
    rationale: str = Field(..., min_length=1, description="Why you think this hypothesis is valid")
    variables: List[str] = Field(default_factory=list, description="Variables involved in the hypothesis")
    data_requirements: List[str] = Field(default_factory=list, description="Data needed to test this hypothesis")
    expected_outcome: str = Field(..., min_length=1, description="Expected result if hypothesis is correct")
    confidence_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in hypothesis (0-1)")
    status: HypothesisStatus = Field(default=HypothesisStatus.DRAFT, description="Current status")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RunStatus(str, Enum):
    """Status of a run"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunSchema(BaseModel):
    """Schema for individual run results"""
    id: Optional[str] = None
    step_id: str = Field(..., description="ID of the analysis step this run belongs to")
    run_number: int = Field(..., ge=1, description="Sequential run number for this step")
    status: RunStatus = Field(default=RunStatus.PENDING, description="Current run status")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for this run")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="Output data from this run")
    results: Dict[str, Any] = Field(default_factory=dict, description="Analysis results")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Run parameters")
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StepSchema(BaseModel):
    """Schema for analysis step with runs"""
    id: Optional[str] = None
    title: str = Field(..., min_length=1, max_length=200, description="Step title")
    description: str = Field(..., min_length=1, description="Step description")
    step_type: str = Field(default="analysis", description="Type of analysis step")
    order: int = Field(..., ge=1, description="Step order in the analysis plan")
    status: RunStatus = Field(default=RunStatus.PENDING, description="Current step status")
    required_data: List[str] = Field(default_factory=list, description="Required data inputs")
    output_schema: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    runs: List[RunSchema] = Field(default_factory=list, description="Runs for this step")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def get_latest_run(self) -> Optional[RunSchema]:
        """Get the most recent run for this step"""
        if not self.runs:
            return None
        return max(self.runs, key=lambda run: run.run_number)

    def get_successful_runs(self) -> List[RunSchema]:
        """Get all successful runs for this step"""
        return [run for run in self.runs if run.status == RunStatus.COMPLETED]

    def get_failed_runs(self) -> List[RunSchema]:
        """Get all failed runs for this step"""
        return [run for run in self.runs if run.status == RunStatus.FAILED]


class SessionStatus(str, Enum):
    """Status of a user session"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"


class SessionData(BaseModel):
    """Schema for user session information"""
    id: Optional[str] = None
    user_id: Optional[str] = None
    session_name: str = Field(..., min_length=1, max_length=100, description="Session name")
    description: Optional[str] = None
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="Current session status")
    current_step_id: Optional[str] = None
    current_hypothesis_id: Optional[str] = None
    analysis_plan: Optional[AnalysisPlan] = None
    hypotheses: List[HypothesisSchema] = Field(default_factory=list, description="Hypotheses in this session")
    steps: List[StepSchema] = Field(default_factory=list, description="Analysis steps")
    dataset_summary: Optional[DatasetSummary] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    settings: Dict[str, Any] = Field(default_factory=dict, description="Session-specific settings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def get_current_step(self) -> Optional[StepSchema]:
        """Get the current step being worked on"""
        if not self.current_step_id:
            return None
        return next((step for step in self.steps if step.id == self.current_step_id), None)

    def get_current_hypothesis(self) -> Optional[HypothesisSchema]:
        """Get the current hypothesis being worked on"""
        if not self.current_hypothesis_id:
            return None
        return next((hyp for hyp in self.hypotheses if hyp.id == self.current_hypothesis_id), None)

    def get_completed_steps(self) -> List[StepSchema]:
        """Get all completed steps"""
        return [step for step in self.steps if step.status == RunStatus.COMPLETED]

    def get_pending_steps(self) -> List[StepSchema]:
        """Get all pending steps"""
        return [step for step in self.steps if step.status == RunStatus.PENDING]

    def get_active_hypotheses(self) -> List[HypothesisSchema]:
        """Get all active hypotheses"""
        return [hyp for hyp in self.hypotheses if hyp.status == HypothesisStatus.ACTIVE]


class AnalysisRequest(BaseModel):
    """Schema for analysis requests"""
    session_id: str
    step_id: Optional[str] = None
    hypothesis_id: Optional[str] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=1, ge=1, le=10, description="Request priority (1-10)")


class AnalysisResponse(BaseModel):
    """Schema for analysis responses"""
    request_id: str
    session_id: str
    step_id: str
    run_id: str
    status: RunStatus
    results: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SessionSummary(BaseModel):
    """Schema for session summary information"""
    id: str
    session_name: str
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    hypothesis_count: int
    step_count: int
    completed_steps: int
    current_step: Optional[str] = None

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }