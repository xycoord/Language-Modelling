import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

def atomic_save_json(filepath: str | Path, data: Dict[str, Any]) -> None:
    """Atomically save data as JSON."""
    if not isinstance(filepath, (str, Path)):
        raise TypeError("filepath must be a string or Path")
    
    filepath = Path(filepath)
    
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=filepath.parent,
        delete=False,
        suffix='.tmp'
    ) as temp_file:
        temp_path = temp_file.name
        
        try:
            json.dump(data, temp_file, indent=2, ensure_ascii=False)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        except Exception:
            os.unlink(temp_path)
            raise
    
    try:
        os.replace(temp_path, filepath)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise

def safe_load_json(filepath: str | Path) -> Dict[str, Any]:
    """Load JSON with helpful error messages."""
    if not isinstance(filepath, (str, Path)):
        raise TypeError("filepath must be a string or Path")
    
    filepath = Path(filepath)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {filepath}: {e}")