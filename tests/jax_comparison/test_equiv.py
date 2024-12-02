"""
Compare torch and jax version
"""
import torch
from elk_torch.algs.algs import quasi_deer_torch
import numpy as np
import pytest

import jax
from elk.algs.elk import elk_alg
import jax.numpy as jnp

# Device selection logic
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Metal Performance Shaders on macOS
elif torch.cuda.is_available():
    device = torch.device("cuda")  # CUDA on NVIDIA GPUs
else:
    device = torch.device("cpu")  # Fallback to CPU

def sample_fxn_jax(state, input):
  """
  Applies a simple transformation on the `state` tensor using a rotation matrix `A` 
  and then a non-linear activation function (tanh).

  Args:
    state (jnp.ndarray): The input state vector or batch of state vectors (B, D).
    input: Unused argument in this function.

  Returns:
    jnp.ndarray: The transformed state after applying the matrix and tanh activation.
  """
  # Small rotation angle
  theta = 0.01  
  # Define a 2D rotation matrix based on theta
  A = jnp.array([[jnp.cos(theta), jnp.sin(theta)], [-jnp.sin(theta), jnp.cos(theta)]])
  # Apply the rotation matrix and tanh activation
  return jnp.tanh(A @ state)

def sample_diag_deriv_jax(state, input):
  """
  Computes the diagonal of the Jacobian matrix for `test_fxn_jax` with respect to `state`.

  Args:
    state (jnp.ndarray): The input state vector or batch of state vectors (B, D).
    input: Unused argument in this function.

  Returns:
    jnp.ndarray: The diagonal of the Jacobian matrix (B, D).
  """
  # Use `jax.jacrev` to compute the Jacobian of `test_fxn_jax` with respect to `state`
  jacobian = jax.jacrev(test_fxn_jax)(state, input)
  # Extract and return the diagonal entries of the Jacobian
  return jnp.diag(jacobian)

# torch fxns
def sample_fxn_torch(input,state):
  """
  Applies a simple transformation on the `state` tensor using a rotation matrix `A` 
 and then a non-linear activation function (tanh).
 y = tanh(A @ x)

 Args:
    state (torch.Tensor): The input state vector or batch of state vectors (B, D).
    input: Unused argument in this function.

  Returns:
    torch.Tensor: The transformed state after applying the matrix and tanh activation.
  """
  device = state.device
  # Small rotation angle
  theta = torch.tensor(0.01, device=device)
  # Define a 2D rotation matrix based on theta
  A = torch.tensor([[torch.cos(theta), torch.sin(theta)], [-torch.sin(theta), torch.cos(theta)]], device=device)
  # Apply the rotation matrix and tanh activation
  # Note: Transpose `A` to align with PyTorch's row-major convention
  return torch.tanh(torch.matmul(state, A.T)) # need to be very careful to handle batch dimension, (B,D)

def sample_diag_deriv_torch(input, state):
  """
  Computes the diagonal of the Jacobian matrix for `test_fxn_torch` with respect to `state`.

  Args:
    state (torch.Tensor): The input state vector or batch of state vectors (B, D).
    input: Unused argument in this function.

  Returns:
    torch.Tensor: The diagonal of the Jacobian matrix (B, D).
  """
  device = state.device
  theta = torch.tensor(0.01, device=device)
  # Define a 2D rotation matrix based on theta
  A = torch.tensor([[torch.cos(theta), torch.sin(theta)], [-torch.sin(theta), torch.cos(theta)]], device=device)
  # Compute activations by applying the rotation matrix
  activations = torch.matmul(state, A.T) # (B,D)

  # Compute the diagonal entries directly using broadcasting
  activations_cosh_squared = torch.cosh(activations) ** 2 # (B,D)
  # Compute the diagonal entries of the derivative
  extra_diag = torch.diag_embed(1 / activations_cosh_squared)  # (B,D, D)
  # Multiply the derivative matrix with the transpose of A to compute diagonal entries
  # Use vmap to apply torch.diag for each batch element
  return torch.func.vmap(torch.diag)(torch.matmul(extra_diag, A.T))


seqlens = [10000]
batch_sizes = [1]
d_inputs = [2]
num_iters = [1] #[1, 2, 5, 10]
dtypes = [torch.float32] #[torch.float32, torch.bfloat16, torch.float16]
#atol = {torch.float32: 1e-7, torch.bfloat16: 1e-1, torch.float16: 12e-3}
atol = {torch.float32: 1e-2, torch.bfloat16: 1e-2, torch.float16: 1e-2}
seeds = [42]


@pytest.mark.parametrize('seqlen', seqlens)
@pytest.mark.parametrize('batch_size', batch_sizes)
@pytest.mark.parametrize('d_input', d_inputs)
@pytest.mark.parametrize('num_iters', num_iters)
@pytest.mark.parametrize('dtype', dtypes)
@pytest.mark.parametrize('seed', seeds)
def test_match_outputs(seqlen, batch_size, d_input, num_iters, dtype, seed):
  torch.manual_seed(seed)

  D = 2
  initial_state_torch = torch.tensor([[1., 0.]], device=device, dtype=dtype)
  states_guess_torch = torch.zeros((batch_size, D, seqlen), device=device, dtype=dtype)
  inputs_torch = torch.zeros(batch_size, d_input, seqlen, device=device, dtype=dtype)


  torch_out = quasi_deer_torch(sample_fxn_torch,
                              sample_diag_deriv_torch,
                              initial_state_torch,
                              states_guess_torch,
                              inputs_torch,
                              num_iters=num_iters)

  initial_state_jax = jnp.array(initial_state_torch.cpu().numpy())
  states_guess_jax = jnp.array(states_guess_torch.cpu().numpy()).transpose(0,2,1) # (B,T,D)
  inputs_jax = jnp.array(inputs_torch.cpu().numpy()).transpose(0,2,1) # (B,T,d_input)

  assert initial_state_jax.shape[0] == states_guess_jax.shape[0] == inputs_jax.shape[0]

  def wrapped_elk_alg(initial_state, states_guess, drivers):
      # output is (B,Niters x T x D)
      return elk_alg(
          f=sample_fxn_jax,
          initial_state=initial_state,
          states_guess=states_guess,
          drivers=drivers,
          quasi=True,
          deer=True,
          num_iters=num_iters
      )

  elk_alg_batch = jax.vmap(wrapped_elk_alg, in_axes=(0, 0, 0), out_axes=0)

  
  jax_out = elk_alg_batch(initial_state_jax,states_guess_jax, inputs_jax)

  assert np.allclose(torch_out.transpose(-1,-2).cpu().numpy().shape, jax_out[:,-1, 1:].shape)
  assert np.allclose(torch_out.transpose(-1,-2).cpu().numpy(), jax_out[:,-1, 1:], atol=atol[dtype], rtol=1e-2)

  #import matplotlib.pyplot as plt
  #plt.plot(torch_out.transpose(-1,-2).cpu().numpy()[0,:,1])
  #plt.plot(jax_out[:,-1, 1:].T[0,:,1])
  #plt.show()
  