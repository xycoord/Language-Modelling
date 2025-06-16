import pytest
import os

from src.datasets.gutenberg_dataset import load_gutenberg_texts


@pytest.fixture
def temp_corpus_dir(tmp_path):
    """Creates a temporary directory for test files"""
    corpus_dir = tmp_path / "test_corpus"
    corpus_dir.mkdir()
    return corpus_dir


@pytest.fixture  
def sample_text_files(temp_corpus_dir):
    """Creates sample .txt files with known content"""
    files = {
        "file_a.txt": "Content of file A",
        "file_b.txt": "Content of file B", 
        "file_c.txt": "Content of file C"
    }
    for filename, content in files.items():
        (temp_corpus_dir / filename).write_text(content, encoding='utf-8')
    return files


@pytest.fixture
def mixed_file_types(temp_corpus_dir):
    """Creates files with different extensions"""
    files = {
        "doc1.txt": "Text content",
        "doc2.md": "Markdown content", 
        "doc3.log": "Log content",
        "readme.txt": "Readme content"
    }
    for filename, content in files.items():
        (temp_corpus_dir / filename).write_text(content, encoding='utf-8')
    return files


@pytest.fixture
def whitespace_files(temp_corpus_dir):
    """Creates files with leading/trailing whitespace"""
    files = {
        "spaced.txt": "  \n  Content with spaces  \n  ",
        "clean.txt": "Clean content"
    }
    for filename, content in files.items():
        (temp_corpus_dir / filename).write_text(content, encoding='utf-8')
    return files


@pytest.fixture
def unsorted_files(temp_corpus_dir):
    """Creates files that test alphabetical sorting"""
    files = {
        "zebra.txt": "Z content",
        "alpha.txt": "A content", 
        "beta.txt": "B content"
    }
    for filename, content in files.items():
        (temp_corpus_dir / filename).write_text(content, encoding='utf-8')
    return files


# Parameterised Tests

@pytest.mark.parametrize("pattern,expected_files", [
    ("*.txt", ["doc1.txt", "readme.txt"]),
    ("*.md", ["doc2.md"]), 
    ("*.log", ["doc3.log"]),
    ("doc*.txt", ["doc1.txt"]),
    ("*", ["doc1.txt", "doc2.md", "doc3.log", "readme.txt"])
])
def test_custom_file_patterns(mixed_file_types, temp_corpus_dir, pattern, expected_files):
    """Test loading files with different glob patterns"""
    texts, filenames = load_gutenberg_texts(str(temp_corpus_dir), pattern)
    
    assert len(texts) == len(expected_files), f"Expected {len(expected_files)} texts, got {len(texts)}"
    assert len(filenames) == len(expected_files), f"Expected {len(expected_files)} filenames, got {len(filenames)}"
    assert sorted(filenames) == sorted(expected_files), f"Expected files {expected_files}, got {filenames}"
    
    for text in texts:
        assert text.strip(), "All returned texts should have non-empty content"


@pytest.mark.parametrize("input_content,expected_output", [
    ("  content  ", "content"),
    ("\n\tcontent\n\t", "content"), 
    ("  \n  content with spaces  \n  ", "content with spaces"),
    ("\r\n\tcontent\r\n", "content"),
    ("content", "content"),  # no whitespace
    ("  multiple\nlines  ", "multiple\nlines")
])
def test_content_stripping(temp_corpus_dir, input_content, expected_output):
    """Test that file content is properly stripped of whitespace"""
    (temp_corpus_dir / "test.txt").write_text(input_content, encoding='utf-8')
    
    texts, filenames = load_gutenberg_texts(str(temp_corpus_dir))
    
    assert len(texts) == 1, "Should load exactly one file"
    assert texts[0] == expected_output, f"Expected '{expected_output}', got '{texts[0]}'"
    assert filenames[0] == "test.txt", f"Expected filename 'test.txt', got '{filenames[0]}'"


@pytest.mark.parametrize("file_content,should_exclude", [
    ("", True),           # truly empty
    ("   ", True),        # whitespace only  
    ("\n\t\n", True),    # whitespace and newlines
    ("\r\n", True),      # carriage returns
    ("a", False),        # minimal content
    ("  a  ", False),    # content with whitespace
])
def test_empty_files_excluded(temp_corpus_dir, file_content, should_exclude):
    """Test that empty files are excluded from results"""
    (temp_corpus_dir / "test.txt").write_text(file_content, encoding='utf-8')
    (temp_corpus_dir / "content.txt").write_text("definite content", encoding='utf-8')
    
    texts, filenames = load_gutenberg_texts(str(temp_corpus_dir))
    
    if should_exclude:
        assert len(texts) == 1, f"Empty file should be excluded, expected 1 text but got {len(texts)}"
        assert filenames == ["content.txt"], f"Should only get content.txt, got {filenames}"
        assert texts[0] == "definite content", f"Expected 'definite content', got '{texts[0]}'"
    else:
        assert len(texts) == 2, f"Non-empty file should be included, expected 2 texts but got {len(texts)}"
        assert "test.txt" in filenames, f"test.txt should be included in {filenames}"
        assert "content.txt" in filenames, f"content.txt should be included in {filenames}"


@pytest.mark.parametrize("filenames,expected_order", [
    (["zebra.txt", "alpha.txt", "beta.txt"], ["alpha.txt", "beta.txt", "zebra.txt"]),
    (["file10.txt", "file2.txt", "file1.txt"], ["file1.txt", "file10.txt", "file2.txt"]),
    (["B.txt", "a.txt", "A.txt"], ["A.txt", "B.txt", "a.txt"])
])
def test_sorted_order_processing(temp_corpus_dir, filenames, expected_order):
    """Test that files are processed in sorted alphabetical order"""
    for i, filename in enumerate(filenames):
        (temp_corpus_dir / filename).write_text(f"Content {i}", encoding='utf-8')
    
    texts, returned_filenames = load_gutenberg_texts(str(temp_corpus_dir))
    
    assert returned_filenames == expected_order, f"Files not in expected order. Expected {expected_order}, got {returned_filenames}"


def test_basic_functionality_default_parameters(sample_text_files, temp_corpus_dir):
    """Test basic loading with default parameters"""
    original_dir = os.getcwd()
    try:
        os.chdir(temp_corpus_dir.parent)
        
        (temp_corpus_dir).rename(temp_corpus_dir.parent / "gutenberg_corpus")
        corpus_path = temp_corpus_dir.parent / "gutenberg_corpus"
        
        texts, filenames = load_gutenberg_texts()
        
        assert len(texts) == 3, f"Expected 3 texts, got {len(texts)}"
        assert len(filenames) == 3, f"Expected 3 filenames, got {len(filenames)}"
        assert sorted(filenames) == ["file_a.txt", "file_b.txt", "file_c.txt"], f"Unexpected filenames: {filenames}"
        
        assert "Content of file A" in texts, "Missing content from file_a.txt"
        assert "Content of file B" in texts, "Missing content from file_b.txt"
        assert "Content of file C" in texts, "Missing content from file_c.txt"
        
    finally:
        os.chdir(original_dir)


def test_custom_directory_parameter(sample_text_files, temp_corpus_dir):
    """Test loading from custom directory"""
    texts, filenames = load_gutenberg_texts(str(temp_corpus_dir))
    
    assert len(texts) == 3, f"Expected 3 texts, got {len(texts)}"
    assert len(filenames) == 3, f"Expected 3 filenames, got {len(filenames)}"
    assert sorted(filenames) == ["file_a.txt", "file_b.txt", "file_c.txt"], f"Unexpected filenames: {filenames}"


def test_single_file_in_directory(temp_corpus_dir):
    """Test loading when directory contains only one matching file"""
    (temp_corpus_dir / "single.txt").write_text("Only file content", encoding='utf-8')
    
    texts, filenames = load_gutenberg_texts(str(temp_corpus_dir))
    
    assert len(texts) == 1, f"Expected 1 text, got {len(texts)}"
    assert len(filenames) == 1, f"Expected 1 filename, got {len(filenames)}"
    assert filenames[0] == "single.txt", f"Expected 'single.txt', got '{filenames[0]}'"
    assert texts[0] == "Only file content", f"Expected 'Only file content', got '{texts[0]}'"


def test_equal_length_lists(sample_text_files, temp_corpus_dir):
    """Test that returned lists are always equal length"""
    texts, filenames = load_gutenberg_texts(str(temp_corpus_dir))
    
    assert len(texts) == len(filenames), f"Lists should be equal length: {len(texts)} texts vs {len(filenames)} filenames"
    assert len(texts) > 0, "Should have loaded some content"


def test_correct_content_mapping(temp_corpus_dir):
    """Test that each filename corresponds to correct content"""
    files = {
        "first.txt": "First file content",
        "second.txt": "Second file content",
        "third.txt": "Third file content"
    }
    
    for filename, content in files.items():
        (temp_corpus_dir / filename).write_text(content, encoding='utf-8')
    
    texts, filenames = load_gutenberg_texts(str(temp_corpus_dir))
    
    content_map = dict(zip(filenames, texts))
    
    assert content_map["first.txt"] == "First file content", f"Wrong content for first.txt: {content_map['first.txt']}"
    assert content_map["second.txt"] == "Second file content", f"Wrong content for second.txt: {content_map['second.txt']}"
    assert content_map["third.txt"] == "Third file content", f"Wrong content for third.txt: {content_map['third.txt']}"


def test_filename_format_basename_only(temp_corpus_dir):
    """Test that only basename is returned, not full path"""
    nested_dir = temp_corpus_dir / "subdir"
    nested_dir.mkdir()
    
    (temp_corpus_dir / "root.txt").write_text("Root content", encoding='utf-8')
    (nested_dir / "nested.txt").write_text("Nested content", encoding='utf-8')
    
    texts, filenames = load_gutenberg_texts(str(temp_corpus_dir))
    
    assert len(filenames) == 1, f"Expected 1 file (glob doesn't recurse), got {len(filenames)}"
    assert filenames[0] == "root.txt", f"Expected 'root.txt', got '{filenames[0]}'"
    
    for filename in filenames:
        assert "/" not in filename, f"Filename should not contain path separators: {filename}"
        assert "\\" not in filename, f"Filename should not contain path separators: {filename}"


def test_nonexistent_directory_exception():
    """Test FileNotFoundError for non-existent directory"""
    with pytest.raises(FileNotFoundError, match="Corpus directory 'nonexistent' not found"):
        load_gutenberg_texts("nonexistent")


def test_no_matching_files_exception(temp_corpus_dir):
    """Test FileNotFoundError when no files match pattern"""
    (temp_corpus_dir / "file.md").write_text("Content", encoding='utf-8')
    (temp_corpus_dir / "readme.log").write_text("Content", encoding='utf-8')
    
    with pytest.raises(FileNotFoundError, match="No files matching '\\*\\.txt' found"):
        load_gutenberg_texts(str(temp_corpus_dir), "*.txt")


def test_mixed_success_failure_scenario(temp_corpus_dir, monkeypatch, capsys):
    """Test handling of mixed readable/unreadable files"""
    files = {
        "good1.txt": "Good content 1",
        "good2.txt": "Good content 2", 
        "bad1.txt": "Will fail to read",
        "bad2.txt": "Will also fail"
    }
    
    for filename, content in files.items():
        (temp_corpus_dir / filename).write_text(content, encoding='utf-8')
    
    # Mock open to selectively fail for files containing "bad"
    original_open = open
    def mock_open_selective(*args, **kwargs):
        file_path = str(args[0])
        if "bad" in file_path:
            raise PermissionError(f"Permission denied: {file_path}")
        return original_open(*args, **kwargs)
    
    monkeypatch.setattr("builtins.open", mock_open_selective)
    
    texts, filenames = load_gutenberg_texts(str(temp_corpus_dir))
    
    assert len(texts) == 2, f"Should only load good files, expected 2 but got {len(texts)}"
    assert len(filenames) == 2, f"Should only get good filenames, expected 2 but got {len(filenames)}"
    assert sorted(filenames) == ["good1.txt", "good2.txt"], f"Expected good files only, got {filenames}"
    assert sorted(texts) == ["Good content 1", "Good content 2"], f"Expected good content only, got {texts}"
    
    captured = capsys.readouterr()
    assert "Error reading bad1.txt" in captured.out, "Should log error for bad1.txt"
    assert "Error reading bad2.txt" in captured.out, "Should log error for bad2.txt"
    assert "Failed to load 2 files" in captured.out, "Should log failed file count"


def test_various_file_read_errors(temp_corpus_dir, monkeypatch):
    """Test graceful handling of different file read error types"""
    # Create test files
    (temp_corpus_dir / "good.txt").write_text("Good content", encoding='utf-8')
    (temp_corpus_dir / "error1.txt").write_text("Content", encoding='utf-8')
    (temp_corpus_dir / "error2.txt").write_text("Content", encoding='utf-8')
    
    # Test different error scenarios
    error_scenarios = [
        ("error1.txt", PermissionError("Permission denied")),
        ("error2.txt", UnicodeDecodeError('utf-8', b'', 0, 1, "Invalid start byte")),
    ]
    
    original_open = open
    
    for error_file, exception in error_scenarios:
        def mock_open_with_error(*args, **kwargs):
            file_path = str(args[0])
            if error_file in file_path:
                raise exception
            return original_open(*args, **kwargs)
        
        monkeypatch.setattr("builtins.open", mock_open_with_error)
        
        # Should handle error gracefully and continue
        texts, filenames = load_gutenberg_texts(str(temp_corpus_dir))
        
        # Should get good file plus any others that didn't error in this iteration
        assert "good.txt" in filenames
        assert len(texts) >= 1  # At least the good file