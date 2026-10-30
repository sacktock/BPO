"""Import surface for the pinned scientific stack.

These stay independent of the training code so a broken dependency install
fails here rather than inside an algorithm test.
"""

import importlib

import pytest

REQUIRED_MODULES = [
    "numpy",
    "jax",
    "jaxlib",
    "tensorflow",
    "tensorflow_probability",
    "flax",
    "optax",
    "gymnasium",
]


@pytest.mark.parametrize("module_name", REQUIRED_MODULES)
def test_required_module_imports(module_name):
    importlib.import_module(module_name)
