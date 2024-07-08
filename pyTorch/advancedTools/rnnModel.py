import torch.nn as nn

class ComplexRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        """
        Initialize the ComplexRNN model.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_dim (int): The dimension of word embeddings.
            hidden_size (int): The size of the hidden state in the RNN.
            output_size (int): The size of the output.
        """
        super(ComplexRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        """
        Forward pass of the ComplexRNN model.

        Args:
            input_sequence (torch.Tensor): Input sequence tensor.

        Returns:
            torch.Tensor: Final output tensor.
        """
        embedded = self.embedding(input_sequence)
        rnn_output, _ = self.rnn(embedded)
        normalized_output = self.layer_norm(rnn_output)
        last_time_step_output = normalized_output[:, -1, :]
        fc1_output = nn.functional.relu(self.fc1(last_time_step_output))
        final_output = self.fc2(fc1_output)
        return final_output
