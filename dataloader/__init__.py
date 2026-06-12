__all__ = ["DataModule"]


def __getattr__(name):
    if name == "DataModule":
        from .data_module import DataModule

        return DataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
