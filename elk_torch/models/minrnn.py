"""
TODO:
    * identity initialization? (Alex Wang suggestion)
"""
import torch
import torch.nn as nn

from elk_torch.algs.algs import quasi_deer_torch


# the MinRNN modules
class MinRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.input_weights = nn.Parameter(
            torch.randn(hidden_size, input_size) / (input_size**0.5)
        )
        self.input_bias = nn.Parameter(torch.zeros(hidden_size))
        self.recurrent_weights = nn.Parameter(
            torch.randn(hidden_size, hidden_size) / (hidden_size**0.5)
        )
        self.U_z = nn.Parameter(
            torch.randn(hidden_size, hidden_size) / (hidden_size**0.5)
        )
        self.b_u = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input, prev_state):
        z = torch.tanh(torch.matmul(input, self.input_weights.T) + self.input_bias)
        u = torch.sigmoid(
            torch.matmul(prev_state, self.recurrent_weights.T)
            + torch.matmul(z, self.U_z.T)
            + self.b_u
        )
        state = u * prev_state + (1 - u) * z
        return state

    def diagonal_derivative(self, input, prev_state):
        z = torch.tanh(torch.matmul(input, self.input_weights.T) + self.input_bias)
        u = torch.sigmoid(
            torch.matmul(prev_state, self.recurrent_weights.T)
            + torch.matmul(z, self.U_z.T)
            + self.b_u
        )
        derivative = (
            u + (prev_state - z) * u * (1 - u) * self.recurrent_weights.diagonal()
        )
        return derivative


class MinRNN(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(MinRNN, self).__init__()
        self.cell = MinRNNCell(hidden_dim, input_dim)

    def forward(self, inputs):
        """
        inputs: (seq_len, batch_size, input_dim)
        outputs: (seq_len, batch_size)
        """
        seq_len, batch_size, _ = inputs.size()
        state = torch.zeros(batch_size, self.cell.hidden_dim).to(inputs.device)  # (B,D)
        outputs = []

        for t in range(seq_len):
            input_t = inputs[t]
            state = self.cell(state, input_t)
            outputs.append(state)

        outputs = torch.stack(outputs, dim=0)
        return outputs

    def parallel_forward(self, inputs, num_iters=10):
        """
        inputs: (seq_len, batch_size, input_dim)
        outputs: (seq_len, batch_size)

        Note: the parallel scan from https://github.com/proger/accelerated-scan
        Takes inputs in shape (B, D, T)
        So some reshaping is needed at the start
        """
        seq_len, batch_size, _ = inputs.size()
        inputs = inputs.permute(1, 2, 0)  # (B, d_input, T)
        hidden_init = torch.zeros(batch_size, self.cell.hidden_dim).to(
            inputs.device
        )  # (B,D)
        states_guess = torch.zeros(batch_size, self.cell.hidden_dim, seq_len).to(
            inputs.device
        )  # (B,D,T), would ideally warm-start though
        outputs = quasi_deer_torch(
            f=self.cell,
            diagonal_derivative=self.cell.diagonal_derivative,
            initial_state=hidden_init,
            states_guess=states_guess,
            drivers=inputs,
            num_iters=num_iters,
        )
        return outputs  # (T,B,D)


class MinRNNClassifier(nn.Module):
    """
    Classifier built on MinRNN for time series classification
    """

    def __init__(self, hidden_dim, input_dim, num_classes=10):
        super(MinRNNClassifier, self).__init__()
        self.rnn = MinRNN(hidden_dim, input_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, parallel=False):
        """
        x: (seq_len, batch_size, input_dim)
        returns: (batch_size, num_classes)
        """
        if parallel:
            rnn_output = self.rnn.parallel_forward(x)
        else:
            rnn_output = self.rnn(x)

        # Use the last output for classification
        last_hidden = rnn_output[-1, :, :]  # (batch_size, hidden_dim)

        # Pass through the classifier
        logits = self.classifier(last_hidden)  # (batch_size, num_classes)

        return logits

    def predict(self, x, parallel=False):
        """
        Convenience method for getting class predictions
        """
        logits = self(x, parallel)
        return torch.argmax(logits, dim=1)
