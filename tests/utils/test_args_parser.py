from src.script_utils.args_parser import ArgsParser
import pytest
import argparse


# =============================================================================
# Shared Test Utilities
# =============================================================================

@pytest.fixture
def basic_config_args():
    """Basic config arguments for testing."""
    return ['--config', 'test_config.yaml']


@pytest.fixture
def config_with_overrides():
    """Config arguments with simple overrides."""
    return [
        '--config', 'test_config.yaml',
        '--override', 'learning_rate=0.01',
        '--override', 'batch_size=32'
    ]

@pytest.fixture
def parser():
    return ArgsParser()


# =============================================================================
# __init__ Tests
# =============================================================================

def test_init_default():
    """Test __init__ with default settings."""
    parser = ArgsParser()
    assert isinstance(parser, ArgsParser)
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.description is None


def test_init_with_description():
    """Test __init__ with custom description."""
    description = "Test parser description"
    parser = ArgsParser(description=description)
    
    assert parser.description == description


# =============================================================================
# parse_config_args Tests  
# =============================================================================

def test_parse_config_args_config_only(basic_config_args, parser):
    """Test parsing with config file only, no overrides."""
    config_path, overrides = parser.parse_config_args(basic_config_args)
    
    assert config_path == 'test_config.yaml'
    assert overrides == {}


def test_parse_config_args_with_simple_overrides(config_with_overrides, parser):
    """Test parsing with simple key=value overrides."""
    config_path, overrides = parser.parse_config_args(config_with_overrides)
    
    assert config_path == 'test_config.yaml'
    assert overrides == {
        'learning_rate': 0.01,
        'batch_size': 32
    }


def test_parse_config_args_with_nested_overrides(parser):
    """Test parsing with nested dot notation overrides."""
    args = [
        '--config', 'test_config.yaml',
        '--override', 'model.embed_dim=512',
        '--override', 'optimizer.lr=0.001'
    ]
    config_path, overrides = parser.parse_config_args(args)
    
    assert config_path == 'test_config.yaml'
    assert overrides == {
        'model': {'embed_dim': 512},
        'optimizer': {'lr': 0.001}
    }


def test_parse_config_args_multiple_overrides(parser):
    """Test parsing with multiple --override arguments."""
    args = [
        '--config', 'test_config.yaml',
        '--override', 'lr=0.01',
        '--override', 'epochs=100',
        '--override', 'debug=true'
    ]
    config_path, overrides = parser.parse_config_args(args)
    
    assert config_path == 'test_config.yaml'
    assert overrides == {
        'lr': 0.01,
        'epochs': 100,
        'debug': True
    }


@pytest.mark.parametrize("input_value,expected_output", [
    # Boolean values
    ("true", True),
    ("false", False),
    ("True", True),
    ("FALSE", False),
    # None/null values
    ("none", None),
    ("null", None),
    ("None", None),
    # Integer values
    ("42", 42),
    ("-5", -5),
    ("0", 0),
    # Float values
    ("3.14", 3.14),
    ("-2.5", -2.5),
    ("0.0", 0.0),
    # String values
    ("hello", "hello"),
    ('"quoted string"', "quoted string"),
    ("", ""),
    # List values
    ("a,b,c", ["a", "b", "c"]),
    ("1,2,3", [1, 2, 3]),
    ("true,false,none", [True, False, None]),
    ("1.5,2.5", [1.5, 2.5]),
])
def test_type_inference_comprehensive(input_value, expected_output, parser):
    """Test comprehensive type inference for override values."""
    args = ['--config', 'test.yaml', '--override', f'key={input_value}']
    
    _, overrides = parser.parse_config_args(args)
    
    assert 'key' in overrides
    assert overrides['key'] == expected_output


def test_nested_keys_multiple_same_parent(parser):
    """Test multiple nested keys under same parent."""
    args = [
        '--config', 'test_config.yaml',
        '--override', 'model.layers=4',
        '--override', 'model.dropout=0.1',
        '--override', 'model.activation=relu'
    ]
    _, overrides = parser.parse_config_args(args)
    
    assert overrides == {
        'model': {
            'layers': 4,
            'dropout': 0.1,
            'activation': 'relu'
        }
    }


def test_nested_keys_mixed_with_flat(parser):
    """Test mix of flat and nested override keys."""
    args = [
        '--config', 'test_config.yaml',
        '--override', 'learning_rate=0.01',
        '--override', 'model.embed_dim=512',
        '--override', 'batch_size=32',
        '--override', 'optimizer.type=adam'
    ]
    _, overrides = parser.parse_config_args(args)
    
    assert overrides == {
        'learning_rate': 0.01,
        'batch_size': 32,
        'model': {'embed_dim': 512},
        'optimizer': {'type': 'adam'}
    }


@pytest.mark.parametrize("args", [
    ([]),  # missing config when required
    (['--config', 'test.yaml', '--override', 'no_equals']),  # no equals sign
    (['--config', 'test.yaml', '--override', '=no_key']),  # equals at start
])
def test_error_cases(args, parser):
    """Test various error conditions that should raise SystemExit."""
    with pytest.raises(SystemExit):
        parser.parse_config_args(args)
    

def test_override_with_empty_value(parser):
    """Test override with empty value should work."""
    args = ['--config', 'test.yaml', '--override', 'empty_key=']
    _, overrides = parser.parse_config_args(args)
    assert overrides == {'empty_key': ''}


def test_multiple_equals_in_override(parser):
    """Test override value containing multiple equals signs."""
    args = [
        '--config', 'test_config.yaml',
        '--override', 'connection_string=user=admin;pass=secret=key'
    ]
    _, overrides = parser.parse_config_args(args)
    
    assert overrides == {
        'connection_string': 'user=admin;pass=secret=key'
    }


def test_require_config_false_no_config_provided():
    """Test that no config is acceptable when require_config=False."""
    parser = ArgsParser(require_config=False)
    args = ['--override', 'test_key=test_value']
    
    config_path, overrides = parser.parse_config_args(args)
    
    assert config_path is None
    assert overrides == {'test_key': 'test_value'}


def test_parse_config_args_with_custom_args(parser):
    """Test parsing with explicitly provided args list."""
    custom_args = ['--config', 'custom.yaml', '--override', 'custom_key=custom_value']
    
    config_path, overrides = parser.parse_config_args(custom_args)
    
    assert config_path == 'custom.yaml'
    assert overrides == {'custom_key': 'custom_value'}