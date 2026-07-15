def __getattr__(name):
    if name == "DataModule":
        from .data_module import DataModule

        return DataModule
    raise AttributeError(name)
