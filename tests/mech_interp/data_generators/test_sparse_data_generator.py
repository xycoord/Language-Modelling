import pytest
import torch
from src.mech_interp.data_generators.sparse_data_generator import SyntheticSparseDataGenerator


@pytest.fixture
def sample_sparsity_1d():
    return torch.tensor([0.0, 0.5, 0.9])


@pytest.fixture
def sample_sparsity_2d():
    return torch.tensor([[0.1, 0.3], [0.7, 0.2]])


@pytest.fixture
def assert_tensor_properties():
    def _assert_properties(tensor, expected_shape, expected_dtype=torch.float32, expected_device=torch.device("cpu")):
        assert isinstance(tensor, torch.Tensor), "Return value must be a torch.Tensor"
        assert tensor.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {tensor.shape}"
        assert tensor.dtype == expected_dtype, f"Dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"
        assert tensor.device == expected_device, f"Device mismatch: expected {expected_device}, got {tensor.device}"
    return _assert_properties


@pytest.mark.parametrize("batch_size,sparsity_tensor", [
    (1, torch.tensor([0.5])),
    (10, torch.tensor([0.0, 0.5, 1.0])),
    (5, torch.tensor([[0.1, 0.3], [0.7, 0.2]])),
])
def test_generator_initialization_valid_inputs(batch_size, sparsity_tensor):
    generator = SyntheticSparseDataGenerator(batch_size, sparsity_tensor)
    
    assert generator.batch_size == batch_size, "batch_size should be stored correctly"
    assert generator.device == torch.device("cpu"), "Default device should be CPU"
    assert generator._data_shape == (batch_size, *sparsity_tensor.shape), "Data shape should include batch dimension"


@pytest.mark.parametrize("batch_size,sparsity_tensor", [
    (0, torch.tensor([0.5])),                    # batch_size at boundary
    (-1, torch.tensor([0.5])),                   # batch_size below valid range
    (10, torch.tensor([1.1])),                   # sparsity above valid range
    (10, torch.tensor([-0.1])),                  # sparsity below valid range
    (10, torch.tensor([0.5, 1.5, 0.2])),         # mixed valid/invalid sparsity
])
def test_generator_initialization_error_conditions(batch_size, sparsity_tensor):
    with pytest.raises(ValueError, match="must be"):
        SyntheticSparseDataGenerator(batch_size, sparsity_tensor)


def test_generate_batch_output_properties_1d(sample_sparsity_1d, assert_tensor_properties):
    generator = SyntheticSparseDataGenerator(5, sample_sparsity_1d)
    batch = generator.generate_batch()
    
    expected_shape = (5, 3)  # batch_size=5, sparsity has 3 elements
    assert_tensor_properties(batch, expected_shape)


def test_generate_batch_output_properties_2d(sample_sparsity_2d, assert_tensor_properties):
    generator = SyntheticSparseDataGenerator(3, sample_sparsity_2d)
    batch = generator.generate_batch()
    
    expected_shape = (3, 2, 2)  # batch_size=3, sparsity is 2x2
    assert_tensor_properties(batch, expected_shape)


def test_generate_batch_value_ranges(sample_sparsity_1d):
    generator = SyntheticSparseDataGenerator(100, sample_sparsity_1d)
    batch = generator.generate_batch()
    
    assert torch.all(batch >= 0.0), "All values must be >= 0"
    assert torch.all(batch <= 1.0), "All values must be <= 1"
    
    non_zero_mask = batch > 0
    if torch.any(non_zero_mask):
        non_zero_values = batch[non_zero_mask]
        assert torch.all(non_zero_values > 0.0), "Non-zero values must be > 0"


def test_device_management_cpu():
    sparsity = torch.tensor([0.5, 0.3])
    generator = SyntheticSparseDataGenerator(10, sparsity)
    
    # Test CPU device (default)
    batch_cpu = generator.generate_batch()
    assert batch_cpu.device == torch.device("cpu"), "Batch should be on CPU by default"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_management_cuda():
    sparsity = torch.tensor([0.5, 0.3])
    generator = SyntheticSparseDataGenerator(10, sparsity)
    
    cuda_device = torch.device("cuda")
    generator_cuda = generator.to(cuda_device)
    
    assert generator_cuda.device == cuda_device, "Generator device should update"
    batch_cuda = generator_cuda.generate_batch()
    assert batch_cuda.device == cuda_device, "Batch should be on CUDA device"
    
    # Test moving back to CPU
    generator_cpu = generator_cuda.to(torch.device("cpu"))
    batch_cpu = generator_cpu.generate_batch()
    assert batch_cpu.device == torch.device("cpu"), "Batch should be back on CPU"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_initialization_cuda():
    sparsity = torch.tensor([0.5, 0.3])
    cuda_device = torch.device("cuda")
    generator = SyntheticSparseDataGenerator(10, sparsity, device=cuda_device)
    
    assert generator.device == cuda_device, "Generator should use specified device"
    batch = generator.generate_batch()
    assert batch.device == cuda_device, "Batch should be on specified device"


def test_determinism_with_seed():
    sparsity = torch.tensor([0.5, 0.3])
    generator = SyntheticSparseDataGenerator(10, sparsity)
    
    # Generate with same seed should produce identical results
    torch.manual_seed(42)
    batch1 = generator.generate_batch()
    
    torch.manual_seed(42)
    batch2 = generator.generate_batch()
    
    assert torch.equal(batch1, batch2), "Same seed should produce identical batches"


def test_randomness_without_seed():
    sparsity = torch.tensor([0.5])
    generator = SyntheticSparseDataGenerator(100, sparsity)
    
    batch1 = generator.generate_batch()
    batch2 = generator.generate_batch()
    
    # Very unlikely to be identical without seeding (though theoretically possible)
    assert not torch.equal(batch1, batch2), "Different calls should produce different results"


@pytest.mark.parametrize("sparsity_value", [0.0, 0.99])
def test_edge_case_sparsity_values(sparsity_value):
    sparsity = torch.tensor([sparsity_value])
    generator = SyntheticSparseDataGenerator(10, sparsity)
    batch = generator.generate_batch()
    
    assert torch.all(batch >= 0.0), "All values must be >= 0"
    assert torch.all(batch <= 1.0), "All values must be <= 1"