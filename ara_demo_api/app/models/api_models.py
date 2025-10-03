from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


# Generic Response Wrapper
class ApiResponse(BaseModel):
    """Generic API response wrapper"""
    success: bool = Field(default=True, description="Whether the request was successful")
    message: str = Field(default="", description="Response message")
    data: Optional[Any] = Field(default=None, description="Response data")
    error: Optional[str] = Field(default=None, description="Error message if any")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# File Upload Models
class UploadResponse(BaseModel):
    """Response after file upload"""
    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    message: str = Field(default="File uploaded successfully", description="Upload status message")
    file_name: str = Field(..., description="Original filename")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    file_type: Optional[str] = Field(default=None, description="MIME type of the file")
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional file metadata")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FileRemoveRequest(BaseModel):
    """Request to remove a file"""
    file_id: str = Field(..., description="ID of the file to remove")


# Analysis Models
class AnalyzeRequest(BaseModel):
    """Request to analyze uploaded data"""
    file_id: str = Field(..., description="ID of the file to analyze")
    analysis_type: str = Field(default="basic", description="Type of analysis to perform")
    options: Dict[str, Any] = Field(default_factory=dict, description="Analysis options")
    force_refresh: bool = Field(default=False, description="Force re-analysis even if cached")


class AnalyzeResponse(BaseModel):
    """Response from data analysis"""
    file_id: str = Field(..., description="ID of the analyzed file")
    columns: List[Dict[str, Any]] = Field(..., description="Column information")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    row_count: int = Field(..., description="Number of rows in the dataset")
    column_count: int = Field(..., description="Number of columns in the dataset")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    processing_time_seconds: Optional[float] = Field(default=None, description="Time taken to analyze")
    data_types: Dict[str, str] = Field(default_factory=dict, description="Data types for each column")
    missing_values: Dict[str, int] = Field(default_factory=dict, description="Missing values per column")
    sample_data: List[Dict[str, Any]] = Field(default_factory=list, description="Sample rows from the dataset")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Hypothesis Models
class HypothesisCreateRequest(BaseModel):
    """Request to create a new hypothesis"""
    title: str = Field(..., min_length=1, max_length=200, description="Hypothesis title")
    description: str = Field(..., min_length=1, description="Detailed hypothesis description")
    prediction: Optional[str] = Field(default=None, description="What you expect to find")
    rationale: Optional[str] = Field(default=None, description="Why you think this hypothesis is valid")
    variables: List[str] = Field(default_factory=list, description="Variables involved")
    expected_outcome: Optional[str] = Field(default=None, description="Expected result")
    confidence_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence level (0-1)")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class HypothesisResponse(BaseModel):
    """Response with hypothesis information"""
    hypothesis_id: str = Field(..., description="Unique identifier for the hypothesis")
    title: str = Field(..., description="Hypothesis title")
    description: str = Field(..., description="Hypothesis description")
    status: str = Field(..., description="Current hypothesis status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    confidence_level: float = Field(..., description="Confidence level")
    variables: List[str] = Field(default_factory=list, description="Variables involved")
    tags: List[str] = Field(default_factory=list, description="Tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Plan Models
class GeneratePlanRequest(BaseModel):
    """Request to generate an analysis plan"""
    hypothesis: str = Field(..., description="Hypothesis description")
    data_summary: Dict[str, Any] = Field(..., description="Summary of available data")
    analysis_goals: List[str] = Field(default_factory=list, description="Specific analysis goals")
    constraints: List[str] = Field(default_factory=list, description="Any constraints or limitations")
    preferred_methods: List[str] = Field(default_factory=list, description="Preferred analysis methods")
    max_steps: int = Field(default=10, ge=1, le=20, description="Maximum number of analysis steps")


class GeneratePlanResponse(BaseModel):
    """Response with generated analysis plan"""
    plan_id: str = Field(..., description="Unique identifier for the plan")
    steps: List[Dict[str, Any]] = Field(..., description="Generated analysis steps")
    plan_accepted: bool = Field(default=False, description="Whether the plan was automatically accepted")
    total_steps: int = Field(..., description="Total number of steps in the plan")
    estimated_duration: Optional[str] = Field(default=None, description="Estimated time to complete")
    plan_summary: str = Field(default="", description="Brief summary of the plan")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Plan generation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional plan metadata")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RefinePlanRequest(BaseModel):
    """Request to refine the analysis plan"""
    message: str = Field(..., min_length=1, description="Feedback message for plan refinement")
    specific_step_id: Optional[str] = Field(default=None, description="Specific step to refine (if applicable)")
    refinement_type: str = Field(default="general", description="Type of refinement requested")
    priority: int = Field(default=1, ge=1, le=5, description="Refinement priority (1-5)")


class AcceptPlanRequest(BaseModel):
    """Request to accept the analysis plan"""
    plan_id: str = Field(..., description="ID of the plan to accept")
    notes: Optional[str] = Field(default=None, description="Optional notes about the acceptance")


# Execution Models
class ExecuteStepRequest(BaseModel):
    """Request to execute an analysis step"""
    step_id: str = Field(..., description="ID of the step to execute")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context and parameters")
    input_data: Optional[Dict[str, Any]] = Field(default=None, description="Input data for the step")
    force_execution: bool = Field(default=False, description="Force execution even if step was already run")
    priority: int = Field(default=1, ge=1, le=10, description="Execution priority (1-10)")


class ExecuteStepResponse(BaseModel):
    """Response from step execution"""
    step_id: str = Field(..., description="ID of the executed step")
    run_id: str = Field(..., description="ID of the execution run")
    status: str = Field(..., description="Execution status")
    results: Dict[str, Any] = Field(default_factory=dict, description="Execution results")
    images: List[str] = Field(default_factory=list, description="Generated image URLs or paths")
    code: Optional[str] = Field(default=None, description="Generated or executed code")
    execution_time: float = Field(..., description="Execution time in seconds")
    started_at: datetime = Field(..., description="Execution start timestamp")
    completed_at: datetime = Field(..., description="Execution completion timestamp")
    error_message: Optional[str] = Field(default=None, description="Error message if execution failed")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="Output data from execution")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StepChatRequest(BaseModel):
    """Request for chat interaction with a step"""
    message: str = Field(..., min_length=1, description="Chat message")
    step_id: str = Field(..., description="ID of the step to chat with")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for the chat")
    chat_type: str = Field(default="question", description="Type of chat interaction")


# Report Models
class GenerateReportRequest(BaseModel):
    """Request to generate a final report"""
    selected_run_ids: List[str] = Field(..., description="IDs of runs to include in the report")
    report_format: str = Field(default="html", description="Report format (html, pdf, markdown)")
    include_code: bool = Field(default=True, description="Include generated code in the report")
    include_images: bool = Field(default=True, description="Include generated images in the report")
    include_raw_data: bool = Field(default=False, description="Include raw data in the report")
    template: Optional[str] = Field(default=None, description="Report template to use")
    custom_sections: List[str] = Field(default_factory=list, description="Custom sections to include")


class GenerateReportResponse(BaseModel):
    """Response with generated report"""
    report_id: str = Field(..., description="Unique identifier for the report")
    report: str = Field(..., description="Generated report content")
    citation_count: int = Field(default=0, description="Number of citations in the report")
    report_format: str = Field(..., description="Format of the generated report")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Report generation timestamp")
    file_size: Optional[int] = Field(default=None, description="Report file size in bytes")
    download_url: Optional[str] = Field(default=None, description="URL to download the report")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional report metadata")
    sections: List[str] = Field(default_factory=list, description="Sections included in the report")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Additional utility models
class ErrorResponse(BaseModel):
    """Standard error response"""
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Number of items per page")
    sort_by: Optional[str] = Field(default=None, description="Field to sort by")
    sort_order: str = Field(default="asc", description="Sort order (asc or desc)")


class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")