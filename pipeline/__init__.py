"""Agentic Text-to-SQL pipeline — Trial 12 prompt stack (59.7% EX on BIRD-SQL train)."""

__all__ = ["run_pipeline"]


def __getattr__(name):
    # Lazy so that importing e.g. pipeline.schema does not pull in the network
    # layer (runner -> llm -> requests). `from pipeline import run_pipeline` still works.
    if name == "run_pipeline":
        from .runner import run_pipeline
        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
