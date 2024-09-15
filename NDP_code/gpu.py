import jax
print(jax.default_backend())

def jax_has_gpu():

    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        return True
    except:
        return False

print(jax_has_gpu())

