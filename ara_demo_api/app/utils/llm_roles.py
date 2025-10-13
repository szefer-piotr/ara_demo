class LowerTier():
    DATA_SUMMARY_MODEL = "gpt-4o-mini"  # ✅ Available and supports JSON
    PLAN_GENERATION_MODEL = "gpt-4o-mini"
    PLAN_GENERATION_CHAT_MODEL = "gpt-4o-mini"
    RUN_EXCUTION_MODEL = "gpt-4o-mini"
    RUN_EXCUTION_CHAT_MODEL = "gpt-4o-mini"
    REPORTING_MODEL = "gpt-4o-mini"
    REPORTING_CHAT_MODEL = "gpt-4o-mini"


class HigherTier():
    DATA_SUMMARY_MODEL = "gpt-4o"  # ✅ Available and supports JSON
    PLAN_GENERATION_MODEL = "gpt-4o"
    PLAN_GENERATION_CHAT_MODEL = "gpt-4o"
    RUN_EXCUTION_MODEL = "gpt-4o"
    RUN_EXCUTION_CHAT_MODEL = "gpt-4o"
    REPORTING_MODEL = "gpt-4o"
    REPORTING_CHAT_MODEL = "gpt-4o"


class ModelConfig():
    """Base model configuration"""
    def __init__(self, tier: str = "lower") -> None:
        config = LowerTier if tier == "lower" else HigherTier
        self.data_summary_model = config.DATA_SUMMARY_MODEL
        self.plan_generation_model = config.PLAN_GENERATION_MODEL
        self.plan_generation_chat_model = config.PLAN_GENERATION_CHAT_MODEL
        self.run_execution_model = config.RUN_EXCUTION_MODEL
        self.run_execution_chat_model = config.RUN_EXCUTION_CHAT_MODEL
        self.reporting_model = config.REPORTING_MODEL
        self.reporting_chat_model = config.REPORTING_CHAT_MODEL

