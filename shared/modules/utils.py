import os
import sys

def resource_path(script_file, path):
    if os.path.isabs(path):
        return path
    
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        base_path = os.path.dirname(sys.executable)
        clean_path = path.removeprefix('../').removeprefix('./')
        return os.path.normpath(os.path.join(base_path, clean_path))
    
    else:
        base_path = os.path.dirname(os.path.abspath(script_file))

    full_path = os.path.normpath(os.path.join(base_path, path))
    return full_path