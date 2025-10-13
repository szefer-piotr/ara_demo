import os
import io
import logging
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from openai import OpenAI
from openai.types.responses import (
    Response,
    ResponseCodeInterpreterToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from openai.types.responses.response_output_text import AnnotationContainerFileCitation

from app.config import settings
from app.utils.llm_roles import ModelConfig
from app.utils.prompt_templates import (
    chat_instructions,
    analysis_steps_generation_instructions,

)

logger = logging.getLogger(__name__)

Chunk = Dict[str, Any]


class LLMService:
    "Service for OpenAI interactions"

    def __init__(self, api_key: Optional[str] = None, model_tier: str = "lower"):
        """
        Initialize OpenAI client

        Args:
            api_key: OpenAI API key (defaults to settings)
            model_tier: "lower" or "higher" for model selection
        """
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")

        self.client = OpenAI(api_key=self.api_key)
        self.model_config = ModelConfig(tier=model_tier)

        self.containers: Dict[str, Any] = {} # maps session_id -> container

        logger.info(f"LLM Service initialized with {model_tier} tier models")

    def create_container(
        self,
        session_id: str,
        file_ids: List[str],
        name: Optional[str] = None
    ) -> Any:
        """
        Create a container for code interpreter with file access

        Args:
            session_id: Session identifier
            file_ids: List of OpenAI file IDs to attach
            name: Container name (defaults to session_id)

        Returns:
            Container object
        """
        try:
            container_name = name or f"session-{session_id}"

            container = self.client.containers.create(
                name=container_name,
                file_ids=file_ids,
                expieres_after={
                    "anchor": "last_active_at",
                    "minutes": 60
                }
            )

            self.containers[session_id] = container

            logger.info(f"Created container")
            return container

        except Exception as e:
                    logger.error(f"Failed to create container: {e}")
                    raise
    
    def get_container(self, session_id: str) -> Optional[Any]:
        """Get cached container for session"""
        return self.containers.get(session_id)
    
    def get_or_create_container(
        self,
        session_id: str,
        file_ids: List[str]
    ) -> Any:
        """Get existing container or create new one"""
        container = self.get_container(session_id)
        
        if container is None:
            container = self.create_container(session_id, file_ids)
        
        return container
    
    def delete_container(self, session_id: str) -> bool:
        """Delete container for session"""
        try:
            container = self.containers.get(session_id)
            if container:
                # Note: OpenAI containers auto-expire, manual deletion may not be needed
                self.containers.pop(session_id, None)
                logger.info(f"Removed container for session {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete container: {e}")
            return False

    def upload_file_to_openai(
        self,
        file_content: bytes,
        filename: str,
        purpose: str = "user_data"
    ) -> str:
        """
        Upload file to OpenAI and return file ID
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            purpose: File purpose ("user_data" or "assistants")
            
        Returns:
            OpenAI file ID
        """
        try:
            # Create file buffer
            file_buffer = io.BytesIO(file_content)
            file_buffer.name = filename
            
            # Upload to OpenAI
            openai_file = self.client.files.create(
                file=file_buffer,
                purpose=purpose
            )
            
            logger.info(f"Uploaded file {filename} to OpenAI: {openai_file.id}")
            return openai_file.id
            
        except Exception as e:
            logger.error(f"Failed to upload file to OpenAI: {e}")
            raise
    
    def upload_csv_to_openai(
        self,
        df: pd.DataFrame,
        filename: str = "data.csv"
    ) -> str:
        """
        Upload DataFrame as CSV to OpenAI
        
        Args:
            df: pandas DataFrame
            filename: Desired filename
            
        Returns:
            OpenAI file ID
        """
        try:
            # Convert DataFrame to UTF-8 CSV
            csv_buffer = io.BytesIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8')
            csv_buffer.seek(0)
            
            # Upload to OpenAI
            openai_file = self.client.files.create(
                file=(filename, csv_buffer, 'text/csv'),
                purpose="user_data"
            )
            
            logger.info(f"Uploaded CSV to OpenAI: {openai_file.id}")
            return openai_file.id
            
        except Exception as e:
            logger.error(f"Failed to upload CSV to OpenAI: {e}")
            raise
    
    # ==================== Tool Creation ====================
    
    def create_code_interpreter_tool(
        self,
        container: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Create code interpreter tool configuration"""
        return {
            "type": "code_interpreter",
            "container": container.id if container else "auto"
        }
    
    def create_web_search_tool(self) -> Dict[str, str]:
        """Create web search tool configuration"""
        return {"type": "web_search_preview"}
    
    # ==================== Data Summarization ====================
    
    def summarize_dataframe(
        self,
        df: pd.DataFrame,
        infer_descriptions: bool = True
    ) -> Dict[str, Any]:
        """
        Generate base data summary (no hypothesis required)
        
        This is the first stage - pure data description.
        Can be called immediately after upload.
        
        Args:
            df: pandas DataFrame to summarize
            infer_descriptions: Whether to infer column descriptions using LLM
            
        Returns:
            Dictionary with comprehensive data summary
        """
        try:
            # Generate basic statistical summary
            summary = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": [],
                "dtypes": {},
                "missing_values": {},
                "sample_data": df.head(10).to_dict('records'),
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Column details
            for col in df.columns:
                col_info = {
                    "name": col,
                    "type": str(df[col].dtype),
                    "non_null_count": int(df[col].count()),
                    "null_count": int(df[col].isna().sum()),
                    "null_percentage": round(df[col].isna().sum() / len(df) * 100, 2),
                    "unique_values": int(df[col].nunique())
                }
                
                # Add statistics for numeric columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    if not df[col].isna().all():
                        col_info["statistics"] = {
                            "mean": float(df[col].mean()),
                            "median": float(df[col].median()),
                            "std": float(df[col].std()),
                            "min": float(df[col].min()),
                            "max": float(df[col].max()),
                            "q25": float(df[col].quantile(0.25)),
                            "q75": float(df[col].quantile(0.75))
                        }
                
                # Add value counts for categorical with few unique values
                elif col_info["unique_values"] <= 20:
                    value_counts = df[col].value_counts().head(10).to_dict()
                    col_info["value_counts"] = {str(k): int(v) for k, v in value_counts.items()}
                
                summary["columns"].append(col_info)
                summary["dtypes"][col] = str(df[col].dtype)
                summary["missing_values"][col] = int(df[col].isna().sum())
            
            # Infer column descriptions if requested (no hypothesis needed)
            # if infer_descriptions:
            #     logger.info("Inferring column descriptions...")
            #     column_descriptions = self.infer_column_descriptions(
            #         data_summary=summary,
            #         df=df
            #     )
                
            #     # Add descriptions to columns
            #     for col_info in summary["columns"]:
            #         col_name = col_info["name"]
            #         if col_name in column_descriptions:
            #             col_info["description"] = column_descriptions[col_name]
                
            #     summary["column_descriptions"] = column_descriptions
            
    

            # In summarize_dataframe, around line 273-286
            if infer_descriptions:
                logger.info("Inferring column descriptions...")
                try:
                    column_descriptions = self.infer_column_descriptions(
                        data_summary=summary,
                        df=df
                    )
                    logger.info(f"✅ Got {len(column_descriptions)} descriptions")
                except Exception as e:
                    logger.error(f"❌ Description inference failed: {e}", exc_info=True)
                    column_descriptions = {}
                
                # Add descriptions to columns
                for col_info in summary["columns"]:
                    col_name = col_info["name"]
                    if col_name in column_descriptions:
                        col_info["description"] = column_descriptions[col_name]
                
                summary["column_descriptions"] = column_descriptions
            
            else:
                summary["column_descriptions"] = {}
                
            logger.info(f"Generated summary for dataset: {len(df)} rows, {len(df.columns)} columns")
            return summary

        except Exception as e:
            logger.error(f"Failed to summarize DataFrame: {e}")
            raise


    def enrich_summary_with_hypothesis(
        self,
        data_summary: Dict[str, Any],
        hypothesis: str,
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Enrich existing data summary with hypothesis-specific analysis
        
        Call this AFTER summarize_dataframe() when hypothesis is provided.
        This adds contextual analysis without recomputing base statistics.
        
        Args:
            data_summary: Output from summarize_dataframe()
            hypothesis: Research hypothesis
            df: Optional DataFrame for additional analysis
            
        Returns:
            Enriched summary with hypothesis-specific insights
        """
        try:
            enriched = data_summary.copy()
            
            # Add hypothesis-specific analysis
            hypothesis_analysis = self._analyze_data_for_hypothesis(
                data_summary=data_summary,
                hypothesis=hypothesis,
                df=df
            )
            
            enriched["hypothesis_analysis"] = hypothesis_analysis
            enriched["enriched_at"] = datetime.utcnow().isoformat()
            
            logger.info(f"Enriched summary with hypothesis analysis")
            return enriched
            
        except Exception as e:
            logger.error(f"Failed to enrich summary: {e}")
            raise


    def _analyze_data_for_hypothesis(
        self,
        data_summary: Dict[str, Any],
        hypothesis: str,
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate hypothesis-specific data analysis
        
        Args:
            data_summary: Base data summary
            hypothesis: Research hypothesis
            df: Optional DataFrame for deeper analysis
            
        Returns:
            Dictionary with hypothesis-specific insights
        """
        try:
            # Format data summary for prompt
            data_desc = self._format_data_summary(data_summary)
            
            prompt = f"""## Research Hypothesis
    {hypothesis}

    ## Available Data
    {data_desc}

    ## Task
    Analyze this dataset in the context of the hypothesis and provide:

    1. **Suitability Assessment**: Is this data appropriate for testing the hypothesis? (2-3 sentences)
    2. **Relevant Variables**: Which columns are most relevant? List them with brief explanations.
    3. **Potential Analyses**: What statistical methods or approaches would be suitable?
    4. **Data Quality Concerns**: Any issues with missing data, outliers, or distribution that could affect testing?
    5. **Additional Data Needs**: What data (if any) would strengthen the analysis?

    Respond in JSON format:
    {{
    "suitability": "assessment text",
    "relevant_variables": [
        {{"column": "name", "relevance": "why it matters"}},
        ...
    ],
    "suggested_methods": ["method 1", "method 2", ...],
    "data_quality_concerns": ["concern 1", "concern 2", ...],
    "additional_data_needs": ["need 1", "need 2", ...],
    "overall_confidence": "high/medium/low",
    "confidence_reasoning": "explanation"
    }}"""
            
            response = self.client.chat.completions.create(
                model=self.model_config.data_summary_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data analyst and statistician. Assess data suitability for research hypotheses."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            import json
            analysis = json.loads(response.choices[0].message.content)
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Hypothesis-specific analysis failed: {e}")
            return {
                "suitability": "Analysis unavailable",
                "error": str(e)
            }


    def infer_column_descriptions(
        self,
        data_summary: Dict[str, Any],
        df: Optional[pd.DataFrame] = None,
        batch_size: int = 5
    ) -> Dict[str, str]:
        """
        Infer descriptions for each column using LLM based on column characteristics
        
        Args:
            data_summary: Output from summarize_dataframe()
            df: Optional DataFrame for additional context (sample values)
            batch_size: Number of columns to process in one LLM call
            
        Returns:
            Dictionary mapping column names to descriptions
        """
        try:
            columns = data_summary.get('columns', [])
            if not columns:
                return {}

            column_descriptions = {}

            for i in range(0, len(columns), batch_size):
                batch = columns[i:i+batch_size]
                prompt = self._build_column_description_prompt(
                    batch,
                    data_summary,
                    df
                )

                response = self.client.chat.completions.create(
                    model=self.model_config.data_summary_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a data analyst expert. Infer what each column represents based on its name, data type, and sample values. Be concise and specific."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )

                import json
                batch_description = json.loads(response.choices[0].message.content)

                for col_name, description in batch_description.get('descriptions', {}).items():
                    column_descriptions[col_name] = description

            logger.info(f"Inferred descriptions for {len(column_descriptions)} columns")
            return column_descriptions

        except Exception as e:
            logger.error(f"Failed to infer column descriptions: {e}")
            return {}

    
    def _build_column_description_prompt(
        self,
        columns: List[Dict[str, Any]],
        data_summary: Dict[str, Any],
        df: Optional[pd.DataFrame] = None
    ) -> str:
        """Build prompt for column description inference"""
        
        prompt_parts = []
        prompt_parts.append("Analyze these columns and provide a concise description for each.")
        prompt_parts.append("Consider the column name, data type, value distribution, and sample data.")
        prompt_parts.append("")
        
        for col in columns:
            col_name = col['name']
            prompt_parts.append(f"## Column: {col_name}")
            prompt_parts.append(f"- Data Type: {col['type']}")
            prompt_parts.append(f"- Unique Values: {col['unique_values']}")
            prompt_parts.append(f"- Missing Values: {col['null_count']} / {data_summary.get('row_count', 0)}")
            
            # Add statistics if numeric
            if 'statistics' in col:
                stats = col['statistics']
                if stats['mean'] is not None:
                    prompt_parts.append(f"- Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}")
                    prompt_parts.append(f"- Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
                    prompt_parts.append(f"- Std Dev: {stats['std']:.2f}")
            
            # Add sample values
            if df is not None and col_name in df.columns:
                sample_values = df[col_name].dropna().head(5).tolist()
                prompt_parts.append(f"- Sample Values: {sample_values}")
            
            prompt_parts.append("")
        
        prompt_parts.append("Respond with JSON in this format:")
        prompt_parts.append('{')
        prompt_parts.append('  "descriptions": {')
        for col in columns:
            prompt_parts.append(f'    "{col["name"]}": "brief description here",')
        prompt_parts.append('  }')
        prompt_parts.append('}')
        
        return "\n".join(prompt_parts)

# Global instances for different tiers
llm_service_lower = None
llm_service_higher = None


def get_llm_service(tier: str = "lower") -> LLMService:
    """
    Get or create LLM service instance
    
    Args:
        tier: "lower" or "higher" for model selection
        
    Returns:
        LLMService instance
    """
    global llm_service_lower, llm_service_higher
    
    if tier == "lower":
        if llm_service_lower is None:
            llm_service_lower = LLMService(model_tier="lower")
        return llm_service_lower
    else:
        if llm_service_higher is None:
            llm_service_higher = LLMService(model_tier="higher")
        return llm_service_higher
