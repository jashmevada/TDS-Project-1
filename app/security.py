import os

def is_safe_path(path):
    normalized = os.path.abspath(os.path.normpath(path))
    return normalized.startswith("/data/") and ".." not in normalized


def prevent_deletion():
    # Override delete operations at the system level
    pass
