"""
Implements parallelizing RNN (ungulate) algorithms in pytorch.
Under active development.

Note that we use the parallel scan from https://github.com/proger/accelerated-scan
This scan expects inputs in the form (B,D,T)
Also, the scan requires sequence lengths must be a power of 2 of lengths between 32 and 65536
Currently getting around by padding (but Jax might do this as well)
Another drawback of this scan is that it has to run on GPU
"""

import torch
import numpy as np
from accelerated_scan.warp import (
    scan,
)  # from https://github.com/proger/accelerated-scan


def quasi_deer_torch(
    f,
    diagonal_derivative,
    initial_state,  # (B,D)
    states_guess,  # (B,D, T)
    inputs,  # (B, d_input, T)
    num_iters=10,  # controls number of newton iterations
    k=0.0,  # controls the amount of damping
):
    """
    Currently is quasi-DEER/ELK

    Important note: the signature for f is different than in the jax version. In keeping with pytorch, f(input, state)

    Args:
      f: a forward fxn that takes in an input and a state, and outputs the next state. (following torch.nn.GRUCell convention)
          In the context of a GRU, f is a GRU cell
          In pytorch setting, f should be able to handle the batch dimension
      diagonal_derivative: a forward fxn that takes in an input and a state, and returns a length D diagonal derivative
          In pytorch setting, f should be able to handle the batch dimension
      initial_state: (B,D)
      states_guess, (B, D, T)
      inputs, (B, d_input, T)
      num_iters: number of iterations to run
      k: int, k=0 is quasi-DEER, k is between 0 and 1, nonzero k is quasi-slim-ELK
    Notes:
    - The initial_state is the fixed starting point.

    The structure looks like the following.
    Let h0 be the initial_state (fixed), h[1:T-1] be the states, and e[0:T-2] be the inputs

    Then our graph looks like

    h0 -----> h1 ---> h2 ---> ..... h_{T-1} ----> h_{T}
              ^       ^       ^          ^          ^
              |       |                   |          |
              e0      e1       ..... e_{T-2}      e_{T-1}

    Use the pytorch scan from: https://github.com/proger/accelerated-scan
    This scan expects inputs in the form (B,D,T)
    Note that the RNN standard (for inputs) in pytorch is (T,B,d_input)

    This scan also requires powers of 2, so padding for now...
    """
    B = states_guess.shape[0]
    D = states_guess.shape[1]
    T = states_guess.shape[-1]
    padded_T = int(2 ** np.ceil(np.log2(T)))  # must be a power of 2
    device = states_guess.device

    def step(states):
        """
        states: B,D,T
        """
        # Evaluate f and its Jacobian in parallel across timesteps 1,..,T-1
        fs = torch.func.vmap(f, in_dims=-1, out_dims=-1)(
            inputs[..., 1:], states[..., :-1]
        )  # (B,D,T-1), in your diagram above, the first interacton is f(e1, h1)

        # Compute the As and bs from fs and Jfs
        As = (1.0 - k) * torch.func.vmap(diagonal_derivative, in_dims=-1, out_dims=-1)(
            inputs[..., 1:], states[..., :-1]
        )  # (B, D, T-1)
        bs = fs - As * states[..., :-1]  # (B, D, T-1)

        # initial_state is h0
        b0 = f(inputs[..., 0], initial_state)  # h1 = f(e0, h0), (B,D)
        A0 = torch.zeros_like(As[..., 0])  # (B,D)
        A = torch.cat(
            [A0.unsqueeze(-1), As, torch.ones([B, D, padded_T - T], device=device)],
            dim=-1,
        )  # (B, D, T)
        b = torch.cat(
            [b0.unsqueeze(-1), bs, torch.zeros([B, D, padded_T - T], device=device)],
            dim=-1,
        )  # (B, D, T)

        # parallel asscociative scan
        new_states = scan(A, b)[..., :T]  # (B, D, T)
        new_states.nan_to_num()  # zero out nans, in place modification to be more memory efficient
        return new_states

    deer_traces = []
    for _ in range(num_iters):
        states_guess = step(states_guess)
        deer_traces.append(states_guess)

    return deer_traces[-1]  # (B, D, T)
