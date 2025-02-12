import os

def is_safe_path(file_path: str):
    return file_path.startswith("/data/")

def prevent_deletion():
    # Override delete operations at the system level
    pass
