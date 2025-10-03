import pydantic_settings
import typing as t
from pydantic import Field, field_validator


class Settings(pydantic_settings.BaseSettings):
    # Postgres Configuration
    postgres_host: str = Field(default="localhost", description="Postgres host", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5433, description="Postgres port", env="POSTGRES_PORT")
    postgres_user: str = Field(default="ara_user", description="Postgres user", env="POSTGRES_USER")
    postgres_password: str = Field(default="changeme123", description="Postgres password", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="ara_demo", description="Postgres database", env="POSTGRES_DB")

    # Redis Configuration
    redis_host: str = Field(default="localhost", description="Redis host", env="REDIS_HOST")
    redis_port: int = Field(default=6380, description="Redis port", env="REDIS_PORT")
    redis_password: str = Field(default="myredissecret", description="Redis password", env="REDIS_PASSWORD")

    # Minio Configuration
    minio_host: str = Field(default="localhost", description="Minio host", env="MINIO_HOST")
    minio_port: int = Field(default=9002, description="Minio port", env="MINIO_PORT")
    minio_console_port: int = Field(default=9003, description="Minio console port", env="MINIO_CONSOLE_PORT")
    minio_access_key: str = Field(default="minioadmin", description="Minio access key", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin123", description="Minio secret key", env="MINIO_SECRET_KEY")
    minio_bucket_name: str = Field(default="ara_demo", description="Minio bucket name", env="MINIO_BUCKET_NAME")
    minio_use_ssl: bool = Field(default=False, description="Minio use SSL", env="MINIO_USE_SSL")

    # API Configuration
    app_name: str = Field(default="ARA Demo API", description="Application name", env="APP_NAME")
    api_title: str = Field(default="ARA Demo API", description="API title", env="API_TITLE")
    api_version: str = Field(default="1.0.0", description="API version", env="API_VERSION")
    api_description: str = Field(default="ARA Demo API", description="API description", env="API_DESCRIPTION")
    api_contact_name: str = Field(default="ARA Demo", description="API contact name", env="API_CONTACT_NAME")
    api_contact_email: str = Field(default="szefep@gmail.com", description="API contact email", env="API_CONTACT_EMAIL")
    debug: bool = Field(default=False, description="Debug mode", env="DEBUG")

    # Security Configuration
    secret_key: str = Field(default="secret_key", description="Secret key", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", description="Algorithm", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, description="Access token expire minutes", env="ACCESS_TOKEN_EXPIRE_MINUTES")
    session_expire_minutes: int = Field(default=480, description="Session expire minutes", env="SESSION_EXPIRE_MINUTES")
    
    # CORS Configuration
    allowed_origins: str = Field(default="http://localhost:3000,http://localhost:8000", description="Comma-separated string of allowed origins", env="ALLOWED_ORIGINS")
    api_keys: str = Field(default="", description="Comma-separated list of API keys", env="API_KEYS")

    # OpenAI Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key", env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.5, description="OpenAI temperature", env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=1000, description="OpenAI max tokens", env="OPENAI_MAX_TOKENS")

    # Google/Gemini Configuration
    google_api_key: str = Field(default="", description="Google API key", env="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash-lite", description="Gemini model", env="GEMINI_MODEL")

    # AWS Configuration
    aws_access_key_id: str = Field(default="", description="AWS access key ID", env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", description="AWS secret access key", env="AWS_SECRET_ACCESS_KEY")

    # Application Configuration
    log_level: str = Field(default="INFO", description="Log level", env="LOG_LEVEL")
    max_file_size: int = Field(default=1024 * 1024 * 10, description="Max file size in bytes (10MB)", env="MAX_FILE_SIZE")
    upload_path: str = Field(default="./uploads", description="Upload path", env="UPLOAD_PATH")

    # Additional environment variables that might be present
    database_url: str = Field(default="", description="Database URL", env="DATABASE_URL")
    redis_url: str = Field(default="", description="Redis URL", env="REDIS_URL")
    minio_endpoint: str = Field(default="", description="MinIO endpoint", env="MINIO_ENDPOINT")
    minio_secure: str = Field(default="False", description="MinIO secure", env="MINIO_SECURE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables

    @field_validator("allowed_origins")
    @classmethod
    def parse_allowed_origins(cls, v: str) -> list[str]:
        """Parse allowed origins from comma-separated string"""
        if not v:
            return []
        return [origin.strip() for origin in v.split(',') if origin.strip()]

    @field_validator("api_keys")
    @classmethod
    def parse_api_keys(cls, v: str) -> list[str]:
        """Parse API keys from comma-separated string"""
        if not v:
            return []
        return [key.strip() for key in v.split(',') if key.strip()]
    
    @field_validator("postgres_port", "redis_port", "minio_port", "minio_console_port")
    @classmethod
    def validate_ports(cls, v: int) -> int:
        """Validate ports"""
        if v < 1 or v > 65535:
            raise ValueError(f"Port must be between 1 and 65535. got {v}")
        return v
    
    @field_validator("openai_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature"""  
        if not 0 <= v <= 2:
            raise ValueError(f"Temperature must be between 0 and 2. got {v}")
        return v
    
    def is_valid_api_key(self, key: str) -> bool:
        """Check if API key is valid"""
        if not key:
            return False
        # Parse the api_keys string to list for comparison
        api_keys_list = self.parse_api_keys(self.api_keys)
        return key in api_keys_list
    
    @property
    def computed_database_url(self) -> str:
        """Get the complete database URL."""
        if self.database_url:
            return self.database_url
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def computed_redis_url(self) -> str:
        """Get the complete Redis URL."""
        if self.redis_url:
            return self.redis_url
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"
        else:
            return f"redis://{self.redis_host}:{self.redis_port}/0"
    
    @property
    def computed_minio_endpoint(self) -> str:
        """Get the complete Minio URL."""
        if self.minio_endpoint:
            return self.minio_endpoint
        protocol = "https" if self.minio_use_ssl else "http"
        return f"{protocol}://{self.minio_host}:{self.minio_port}"

settings = Settings()