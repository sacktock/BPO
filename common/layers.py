import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Tuple, Any
from flax.linen import initializers

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

class Flatten(nn.Module):
    """
    Equivalent to PyTorch nn.Flatten() layer.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape((x.shape[0], -1))

class Identity(nn.Module):

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

class NatureCNN(nn.Module):
    grayscale_obs: bool = True
    normalize_images: bool = True
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.lecun_normal()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.grayscale_obs:
            x = jnp.transpose(x, (0, 2, 3, 1))
        else:
            x = jnp.transpose(x, (0, 4, 2, 3, 1))
        if self.normalize_images:
            x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID", kernel_init=self.kernel_init)(x)
        x = self.activation_fn(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID", kernel_init=self.kernel_init)(x)
        x = self.activation_fn(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID", kernel_init=self.kernel_init)(x)
        x = self.activation_fn(x)
        return x.reshape(x.shape[0], -1)