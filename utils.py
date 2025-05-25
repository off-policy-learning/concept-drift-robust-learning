import numpy as np
import random
import jax
import jax.numpy as jnp
from dataclasses import fields
import chex

@chex.dataclass(frozen=True)
class DataInput:
    s: jnp.ndarray
    a: jnp.ndarray
    r: jnp.ndarray
    reward_mat: jnp.ndarray
    a_prob: jnp.ndarray

def set_global_seeds(seed: int):
    """
    Sets global seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    rng_key = jax.random.PRNGKey(seed)
    return rng_key

def subarray_datacls(x: DataInput, subarray_len: int, from_back: bool = False):
    """
    Creates a new instance of a truncated data class. All fields need to be length n.
    """
    d = {}
    n = None
    for f in fields(x):
        f = f.name
        v = getattr(x, f)
        if v is None:
            continue

        if n is None:
            n = v.shape[0]
        assert v.shape[0] == n
        if not from_back:
            d[f] = v[:subarray_len]
        else:
            d[f] = v[len(v) - subarray_len :]
    return type(x)(**d)
