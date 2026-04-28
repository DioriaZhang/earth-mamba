"""
Mamba-3 Util Functions.

Copyright (c) 2025, Dao AI Lab, Goombalab
"""

import triton
import triton.language as tl

# We use PTX approximations instead of triton built-in functions
# to trade off a bit of accuracy for much faster speed.

@triton.jit
def cos_approx(x):
    """
    (Fast) Cosine approximation using PTX inline assembly.

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        Approximate cosine values in float32
    """
    return tl.inline_asm_elementwise(
        "cos.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sin_approx(x):
    """
    (Fast) Sine approximation using PTX inline assembly.

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        Approximate sine values in float32
    """
    return tl.inline_asm_elementwise(
        "sin.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )

@triton.jit
def tanh_approx(x):
    """
    Hyperbolic tangent. Uses exp form so sm_70 (V100) works: PTX ``tanh.approx.f32``
    and Triton ``tl.math.tanh`` both require sm_75+ in practice for this toolchain.
    Clamps x to avoid exp(2x) overflow in fp32.
    """
    x = x.to(tl.float32)
    xc = tl.where(x > 20.0, 20.0, tl.where(x < -20.0, -20.0, x))
    t = tl.exp(2.0 * xc)
    return (t - 1.0) / (t + 1.0)


@triton.jit
def sech2_approx(x):
    """sech^2(x) = 1 - tanh^2(x), same sm_70-safe path as tanh_approx."""
    t = tanh_approx(x)
    return 1.0 - t * t

@triton.jit
def sigmoid_approx(x):
    """
    (Fast) Sigmoid approximation using PTX inline assembly.

    Formula: sigmoid(x) = 0.5 * (1 + tanh(0.5 * x))
    Leverages fast tanh approximation for speed.

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        Approximate sigmoid values in float32
    """
    # tanh_half_x = tl.inline_asm_elementwise(
    #     "tanh.approx.f32 $0, $1;",
    #     constraints="=f,f",
    #     args=[0.5 * x],
    #     dtype=tl.float32,
    #     is_pure=True,
    #     pack=1,
    # )
    # return 0.5 * (1.0 + tanh_half_x)
    # NOTE: We ended up using the built-in sigmoid for better performance, as the PTX approximation was not faster in this case.
    return tl.sigmoid(x)

@triton.jit
def silu(x):
    """
    SiLU (Swish) activation function: x * sigmoid(x).

    Formula: silu(x) = 0.5*x * (1 + tanh(0.5*x)) + 0.5*x.
    Leverages fast tanh_approx for speed.
    
    Args:
        x: Input triton tensor (any shape) in float32
    
    Returns:
        SiLU activation output in float32
    """
    # x_half = 0.5 * x
    # return x_half * tanh_approx(x_half) + x_half
    # NOTE: We ended up using the built-in sigmoid for better performance, as the PTX approximation was not faster in this case.
    return x*tl.sigmoid(x)