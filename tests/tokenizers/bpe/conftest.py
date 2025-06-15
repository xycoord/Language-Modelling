import tempfile
import os
from pathlib import Path
import pytest

@pytest.fixture
def temp_tokenizer_file():
    """Temporary file for tokenizer testing."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    yield Path(temp_path)
    if os.path.exists(temp_path):
        os.unlink(temp_path)