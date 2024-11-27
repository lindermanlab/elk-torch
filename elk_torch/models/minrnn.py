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
    def __init__(self, hidden_size, input_size):
        super(MinRNN, self).__init__()
        self.cell = MinRNNCell(hidden_size, input_size)

    def forward(self, inputs):
        """
        We are going to follow the "batch_first" convention,

        inputs: (batch_size, seq_len, input_size)
        outputs: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = inputs.size()
        state = torch.zeros(batch_size, self.cell.hidden_size).to(inputs.device)  # (B,D)
        outputs = []

        for t in range(seq_len):
            input_t = inputs[t]
            state = self.cell(state, input_t) # (B,D)
            outputs.append(state)

        outputs = torch.stack(outputs, dim=1) # (B,T,D)
        return outputs

    def parallel_forward(self, inputs, num_iters=10):
        """
        inputs: (batch_size, seq_len, input_size)
        outputs: (batch_size, seq_len, hidden_size)

        Note: the parallel scan from https://github.com/proger/accelerated-scan
        Takes inputs in shape (B, D, T)
        So some reshaping is needed at the start
        """
        batch_size, seq_len, _ = inputs.size()
        inputs = inputs.permute(0, 2, 1)  # (B, d_input, T)
        hidden_init = torch.zeros(batch_size, self.cell.hidden_size).to(
            inputs.device
        )  # (B,D)
        states_guess = torch.zeros(batch_size, self.cell.hidden_size, seq_len).to(
            inputs.device
        )  # (B,D,T), would ideally warm-start though
        outputs = quasi_deer_torch(
            f=self.cell,
            diagonal_derivative=self.cell.diagonal_derivative,
            initial_state=hidden_init, # (B,D)
            states_guess=states_guess, # (B,D,T)
            inputs=inputs, # (B,d_input,T)
            num_iters=num_iters,
        ) # (B,D,T)
        return outputs.permute(0, 2, 1) # (B,T,D)


class MinRNNClassifier(nn.Module):
    """
    Classifier built on MinRNN for time series classification

    Note: follows batch first convention
    """

    def __init__(self, hidden_size, input_size, num_classes=10):
        super(MinRNNClassifier, self).__init__()
        self.rnn = MinRNN(hidden_size, input_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, parallel=False):
        """
        x: (batch_size, seq_len, input_size)
        returns: (batch_size, num_classes)
        """
        if parallel:
            rnn_output = self.rnn.parallel_forward(x) # (batch_size, seq_len, hidden_size)
        else:
            rnn_output = self.rnn(x) # (batch_size, seq_len, hidden_size)

        # Use the last output for classification
        last_hidden = rnn_output[:, -1, :]  # (batch_size, hidden_size)

        # Pass through the classifier
        logits = self.classifier(last_hidden)  # (batch_size, num_classes)

        return logits

    def predict(self, x, parallel=False):
        """
        Convenience method for getting class predictions
        Be careful about the parallel flag!
        """
        logits = self(x, parallel)
        return torch.argmax(logits, dim=1)
