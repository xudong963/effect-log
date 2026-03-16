"""Framework middleware for effect-log.

Each middleware module provides integration with a specific agent framework.
Framework dependencies are optional — import errors are raised only when
you actually try to use the middleware.

Available middleware:
    - effect_log.middleware.langgraph      — LangGraph / LangChain
    - effect_log.middleware.openai_agents  — OpenAI Agents SDK
    - effect_log.middleware.crewai         — CrewAI
    - effect_log.middleware.pydantic_ai    — Pydantic AI
    - effect_log.middleware.anthropic      — Anthropic Claude API
"""
