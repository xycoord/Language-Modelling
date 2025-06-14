import pytest
import yaml
import re
from src.utils.config_loader import Config


# =============================================================================
# Shared Test Utilities
# =============================================================================

@pytest.fixture
def base_config_dict():
    """Complete valid config that can be customized per test."""
    return {
        'experiment_name': 'test_exp',
        'run_id': 'test_run_123',
        'model_type': 'transformer',
        'model_config': {'embed_dim': 128, 'num_heads': 8},
        'seed': 42,
        'learning_rate': 0.01,
        'batch_size': 32,
        'block_size': 128,
        'vocab_size': 1000,
        'epochs': 10,
        'eval_interval': 100,
        'example_interval': 500,
        'train_split': 0.8,
        'data_dir': '/data',
        'output_dir': '/output'
    }


def create_config_file(tmp_path, filename, config_dict):
    """Helper to create YAML config files from dictionaries."""
    config_file = tmp_path / filename
    config_file.write_text(yaml.dump(config_dict))
    return str(config_file)


def create_invalid_yaml_file(tmp_path, filename):
    """Helper to create invalid YAML that will cause parsing errors."""
    config_file = tmp_path / filename
    # This creates genuinely invalid YAML - unclosed bracket at end of file
    config_file.write_text("experiment_name: test\nmodel_config: [")
    return str(config_file)


@pytest.fixture
def config_factory(tmp_path):
    """Factory fixture to create config files on demand."""
    def _create_config(filename, config_dict):
        return create_config_file(tmp_path, filename, config_dict)
    return _create_config


# =============================================================================
# from_file Tests
# =============================================================================

def test_from_file_basic_functionality(tmp_path, base_config_dict):
    """Test loading a simple config without inheritance."""
    config_file = create_config_file(tmp_path, "config.yaml", base_config_dict)
    
    config = Config.from_file(config_file)
    
    assert config.experiment_name == 'test_exp'
    assert config.model_type == 'transformer'
    assert config.learning_rate == 0.01
    assert config.model_config == {'embed_dim': 128, 'num_heads': 8}


def test_from_file_overrides(tmp_path, base_config_dict):
    """Test applying overrides to loaded config."""
    config_file = create_config_file(tmp_path, "config.yaml", base_config_dict)
    overrides = {
        'learning_rate': 0.001,
        'batch_size': 64,
        'model_config': {'embed_dim': 256}
    }
    
    config = Config.from_file(config_file, overrides)
    
    assert config.learning_rate == 0.001
    assert config.batch_size == 64
    # Model config should be merged, not replaced
    assert config.model_config == {'embed_dim': 256, 'num_heads': 8}


@pytest.mark.parametrize("has_run_id", [True, False])
def test_from_file_run_id_handling(tmp_path, base_config_dict, has_run_id):
    """Test run_id generation vs preservation."""
    if not has_run_id:
        base_config_dict.pop('run_id')
    
    config_file = create_config_file(tmp_path, "config.yaml", base_config_dict)
    config = Config.from_file(config_file)
    
    if has_run_id:
        assert config.run_id == 'test_run_123'
    else:
        assert re.match(r'\d{8}_\d{6}_[a-f0-9]{8}', config.run_id)


def test_from_file_single_inheritance(config_factory):
    """Test config inheriting from one parent."""
    parent_config = {
        'experiment_name': 'test_exp',
        'model_type': 'transformer',
        'model_config': {'embed_dim': 64},
        'seed': 42,
        'learning_rate': 0.001,
        'batch_size': 16,
        'block_size': 128,
        'vocab_size': 1000,
        'data_dir': '/data',
        'output_dir': '/output'
    }
    parent_file = config_factory("parent.yaml", parent_config)
    
    child_config = {
        'inherits': [parent_file],
        'run_id': 'child_run_456',
        'learning_rate': 0.01,  # Override parent
        'epochs': 20,           # Add missing field
        'eval_interval': 100,   # Add missing field
        'example_interval': 500, # Add missing field
        'train_split': 0.8      # Add missing field
    }
    child_file = config_factory("child.yaml", child_config)
    
    config = Config.from_file(child_file)
    
    assert config.learning_rate == 0.01  # Child overrides parent
    assert config.batch_size == 16  # Inherited from parent
    assert config.epochs == 20  # From child
    assert config.experiment_name == 'test_exp'  # Inherited from parent
    assert config.model_config == {'embed_dim': 64}  # Inherited from parent


def test_from_file_multiple_inheritance(config_factory):
    """Test config inheriting from multiple parents (later parents override earlier ones)."""
    parent1_config = {
        'experiment_name': 'test_exp',
        'model_type': 'transformer',
        'learning_rate': 0.001,
        'batch_size': 16,
        'model_config': {'embed_dim': 64},
        'seed': 42,
        'data_dir': '/data'
    }
    parent1_file = config_factory("parent1.yaml", parent1_config)
    
    parent2_config = {
        'learning_rate': 0.002,  # Will override parent1
        'block_size': 128,
        'vocab_size': 1000,
        'epochs': 50,
        'model_config': {'num_heads': 4},  # Will merge with parent1
        'output_dir': '/output'
    }
    parent2_file = config_factory("parent2.yaml", parent2_config)
    
    child_config = {
        'inherits': [parent1_file, parent2_file],
        'run_id': 'child_run_789',
        'batch_size': 32,        # Child overrides both parents
        'eval_interval': 100,    # Add missing field
        'example_interval': 500, # Add missing field
        'train_split': 0.8       # Add missing field
    }
    child_file = config_factory("child.yaml", child_config)
    
    config = Config.from_file(child_file)
    
    assert config.learning_rate == 0.002  # parent2 overrides parent1
    assert config.batch_size == 32  # Child overrides both parents
    assert config.epochs == 50  # From parent2
    assert config.experiment_name == 'test_exp'  # From parent1
    # Model config merged from both parents
    assert config.model_config == {'embed_dim': 64, 'num_heads': 4}


def test_from_file_nested_inheritance(config_factory):
    """Test inheritance where parent also inherits (recursive)."""
    grandparent_config = {
        'experiment_name': 'test_exp',
        'model_type': 'transformer',
        'learning_rate': 0.0001,
        'model_config': {'embed_dim': 32},
        'seed': 42,
        'data_dir': '/data'
    }
    grandparent_file = config_factory("grandparent.yaml", grandparent_config)
    
    parent_config = {
        'inherits': [grandparent_file],
        'batch_size': 16,
        'block_size': 128,
        'vocab_size': 1000,
        'model_config': {'num_heads': 2},  # Merges with grandparent
        'output_dir': '/output'
    }
    parent_file = config_factory("parent.yaml", parent_config)
    
    child_config = {
        'inherits': [parent_file],
        'run_id': 'nested_run_123',
        'learning_rate': 0.01,   # Child overrides grandparent
        'epochs': 10,            # Add missing field
        'eval_interval': 100,    # Add missing field
        'example_interval': 500, # Add missing field
        'train_split': 0.8       # Add missing field
    }
    child_file = config_factory("child.yaml", child_config)
    
    config = Config.from_file(child_file)
    
    assert config.learning_rate == 0.01  # Child overrides grandparent
    assert config.batch_size == 16  # From parent
    assert config.experiment_name == 'test_exp'  # From grandparent
    # Model config merged through inheritance chain
    assert config.model_config == {'embed_dim': 32, 'num_heads': 2}


def test_from_file_inheritance_with_overrides(config_factory):
    """Test inheritance combined with runtime overrides."""
    parent_config = {
        'experiment_name': 'test_exp',
        'model_type': 'transformer',
        'learning_rate': 0.001,
        'model_config': {'embed_dim': 64},
        'seed': 42,
        'batch_size': 32,
        'block_size': 128,
        'vocab_size': 1000,
        'data_dir': '/data',
        'output_dir': '/output'
    }
    parent_file = config_factory("parent.yaml", parent_config)
    
    child_config = {
        'inherits': [parent_file],
        'run_id': 'override_run_456',
        'batch_size': 16,        # Child overrides parent
        'epochs': 10,            # Add missing field
        'eval_interval': 100,    # Add missing field
        'example_interval': 500, # Add missing field
        'train_split': 0.8       # Add missing field
    }
    child_file = config_factory("child.yaml", child_config)
    
    overrides = {
        'learning_rate': 0.005,
        'model_config': {'num_heads': 12}
    }
    
    config = Config.from_file(child_file, overrides)
    
    assert config.learning_rate == 0.005  # Override wins over parent
    assert config.batch_size == 16  # From child config
    assert config.experiment_name == 'test_exp'  # From parent
    # Model config merged: parent + override
    assert config.model_config == {'embed_dim': 64, 'num_heads': 12}


@pytest.mark.parametrize("error_type,setup_func,expected_exception,expected_message", [
    (
        "file_not_found",
        lambda tmp_path: "nonexistent.yaml",
        FileNotFoundError,
        "Config file not found"
    ),
    (
        "invalid_yaml", 
        lambda tmp_path: create_invalid_yaml_file(tmp_path, "bad.yaml"),
        ValueError,
        "Invalid YAML"
    ),
    (
        "unknown_model_type",
        lambda tmp_path, base_config: create_config_file(
            tmp_path, "bad_model.yaml", {**base_config, 'model_type': 'unknown'}
        ),
        ValueError,
        "Unknown model type"
    )
])
def test_from_file_error_cases(tmp_path, base_config_dict, error_type, setup_func, expected_exception, expected_message):
    """Test various error conditions."""
    if error_type == "unknown_model_type":
        config_file = setup_func(tmp_path, base_config_dict)
    else:
        config_file = setup_func(tmp_path)
    
    with pytest.raises(expected_exception, match=expected_message):
        Config.from_file(config_file)


# =============================================================================
# __post_init__ Tests
# =============================================================================

@pytest.mark.parametrize("model_type,expected_config_type", [
    ("transformer", "TransformerConfig"),
    ("bigram", "BigramConfig")
])
def test_post_init_model_config_creation(tmp_path, base_config_dict, model_type, expected_config_type):
    """Test creation of typed model config objects."""
    test_config = {**base_config_dict, 'model_type': model_type}
    
    if model_type == 'bigram':
        test_config['model_config'] = {}
    else:
        test_config['model_config'] = {'embed_dim': 128, 'num_heads': 8}
    
    config_file = create_config_file(tmp_path, "config.yaml", test_config)
    config = Config.from_file(config_file)
    
    assert config.model_config_typed.__class__.__name__ == expected_config_type
    assert config.model_config_typed.vocab_size == config.vocab_size
    if model_type == 'transformer':
        assert config.model_config_typed.block_size == config.block_size


def test_post_init_unknown_model_type_error(tmp_path, base_config_dict):
    """Test error handling for unknown model type in __post_init__."""
    test_config = {**base_config_dict, 'model_type': 'invalid_model'}
    config_file = create_config_file(tmp_path, "config.yaml", test_config)
    
    with pytest.raises(ValueError, match="Unknown model type"):
        Config.from_file(config_file)


def test_post_init_flat_config_creation(tmp_path, base_config_dict):
    """Test creation of flattened config for logging."""
    config_file = create_config_file(tmp_path, "config.yaml", base_config_dict)
    config = Config.from_file(config_file)
    
    assert 'model_config' not in config.flat_config
    assert 'model_embed_dim' in config.flat_config
    assert 'model_num_heads' in config.flat_config
    assert config.flat_config['model_embed_dim'] == 128
    assert config.flat_config['learning_rate'] == 0.01
    assert config.flat_config['experiment_name'] == 'test_exp'