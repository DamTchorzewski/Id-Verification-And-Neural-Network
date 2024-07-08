import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from wordPreprocessing import preprocess_data
from rnnModel import ComplexRNN  # Assuming you have a ComplexRNN class defined in RNN_Model.py
from trainedModel import predict_next_words
import torch.optim.lr_scheduler as lr_scheduler
import os
import re
import matplotlib.pyplot as plt

def get_absolute_path(relative_path):
    """
    Get the absolute path given a relative path.

    Args:
        relative_path (str): The relative path.

    Returns:
        str: The absolute path.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, relative_path)

def load_vocabulary(file_path):
    """
    Load vocabulary from a file.

    Args:
        file_path (str): The path to the vocabulary file.

    Returns:
        dict: The vocabulary.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return {}
    vocab = {}
    with open(file_path, 'r', encoding='utf-8') as vocab_file:
        for line in vocab_file:
            if ': ' in line:
                token, idx = line.strip().split(': ', 1)
                vocab[token] = int(idx)
    print(f"Vocabulary loaded successfully. Size: {len(vocab)}")
    return vocab

def load_word_pairs(file_path, vocab_size):
    """
    Load word pairs from a file.

    Args:
        file_path (str): The path to the word pairs file.
        vocab_size (int): The size of the vocabulary.

    Returns:
        tuple: A tuple containing input sequences and output words.
    """
    input_sequences, output_words = [], []

    with open(file_path, 'r', encoding='utf-8') as word_pairs_file:
        for line in word_pairs_file:
            match = re.match(r"Input Sequence: \[([\d, ]+)\], Output Word: (\d+)", line.strip())
            if match:
                try:
                    input_sequence_str = match.group(1)
                    output_word = int(match.group(2))

                    input_sequence = [int(idx) for idx in input_sequence_str.split(',')]

                    # Check if all indices in the input sequence are within the vocabulary size
                    if all(0 <= idx < vocab_size for idx in input_sequence):
                        input_sequences.append(input_sequence)
                        output_words.append(output_word)
                except Exception as e:
                    print(f"Error processing line: {line}")
                    print(f"Exception: {e}")
                    continue

    if len(input_sequences) > 0:
        print(f"Word pairs loaded successfully. Size: {len(input_sequences)}")
        print(f"Example input sequence: {input_sequences[0]}")
        print(f"Example output word: {output_words[0]}")
    else:
        print("Error: Dataset is empty.")

    return input_sequences, output_words

def calculate_accuracy(outputs, labels):
    """
    Calculate the accuracy of predictions.

    Args:
        outputs (torch.Tensor): The model outputs.
        labels (torch.Tensor): The target labels.

    Returns:
        float: The accuracy.
    """
    _, predicted = torch.max(outputs, 1)
    correct = torch.sum(predicted == labels).item()
    total = labels.size(0)
    accuracy = (correct / total) * 100  # Convert accuracy to percentage
    return accuracy

def train_model(train_dataloader, rnn_model, criterion, optimizer, scheduler, vocab_size, num_epochs):
    """
    Train the RNN model.

    Args:
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        rnn_model (torch.nn.Module): The RNN model.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        vocab_size (int): The size of the vocabulary.
        num_epochs (int): The number of epochs.

    Returns:
        tuple: A tuple containing lists of training losses, training accuracies, and the vocabulary size.
    """
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        # Training
        rnn_model.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = rnn_model(inputs)

            # Ensure labels are within the range [0, vocab_size)
            labels = labels % vocab_size

            # Calculate loss using class indexes directly
            loss = criterion(outputs, labels)
            loss.backward()

            # Clip gradients during training
            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), max_norm=1.0)

            optimizer.step()

            total_train_loss += loss.item()
            total_train_correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()  # Use torch.argmax here
            total_train_samples += labels.size(0)

        # Calculate accuracy after processing all batches in the training set
        average_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = total_train_correct / total_train_samples
        train_losses.append(average_train_loss)
        train_accuracies.append(train_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {average_train_loss}, Train Accuracy: {train_accuracy * 100:.2f}%')

        # Pass the validation metric value to the scheduler
        scheduler.step(average_train_loss)

    return train_losses, train_accuracies, vocab_size

def main():
    # Model file name
    model_file_name = 'generate_sentense.txt'  # Please change the file name for the one available in the base folder

    # Preprocess data and get file paths
    (vocab_file_path, train_word_pairs_file_path, test_word_pairs_file_path, sequence_length) = preprocess_data(
        model_file_name)

    # Load vocabulary
    vocab = load_vocabulary(vocab_file_path)
    vocab_size = len(vocab)

    # Define hyperparameters
    embedding_dim = 256
    hidden_size = 256
    num_epochs = 40
    learning_rate = 0.0001

    model_file_name_trained = f'rnn_model_{model_file_name}_{sequence_length}.pth'

    rnn_model = ComplexRNN(vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size,
                           output_size=vocab_size).to('cuda')

    # Check if the model file exists
    if os.path.exists(model_file_name_trained):
        # If the model file exists, load the model
        rnn_model.load_state_dict(torch.load(model_file_name_trained))
        print(f"Loaded pre-trained model from {model_file_name_trained}")

    # Load training word pairs
    train_input_sequences, train_output_words = load_word_pairs(train_word_pairs_file_path, vocab_size)

    # Convert to PyTorch tensors
    train_input_sequences_tensor = torch.tensor(train_input_sequences, dtype=torch.long)
    train_output_words_tensor = torch.tensor(train_output_words, dtype=torch.long)

    # Move tensors to GPU
    train_input_sequences_tensor, train_output_words_tensor = (train_input_sequences_tensor.to('cuda'),
                                                               train_output_words_tensor.to('cuda'))

    # Initialize and train the model
    print(rnn_model)  # Display model architecture
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn_model.parameters(), learning_rate)

    # Example of reducing learning rate during training
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True)

    # Create DataLoader for training
    train_dataset = TensorDataset(train_input_sequences_tensor, train_output_words_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=False)

    # Train the model
    train_losses, train_accuracies, vocab_size = train_model(
        train_dataloader, rnn_model, criterion, optimizer, scheduler, vocab_size, num_epochs)

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Save plots
    plt.savefig(f'training_plots_{model_file_name}_{sequence_length}.png')

    plt.show()

    # Save the trained model
    trained_model_filename = f'rnn_model_{model_file_name}_{sequence_length}.pth'
    torch.save(rnn_model.state_dict(), trained_model_filename)
    print(f"Saved trained model to {trained_model_filename}")

    # Load training word pairs
    test_input_sequences, test_output_words = load_word_pairs(test_word_pairs_file_path, vocab_size)

    # Convert to PyTorch tensors
    test_input_sequences_tensor = torch.tensor(test_input_sequences, dtype=torch.long)
    test_output_words_tensor = torch.tensor(test_output_words, dtype=torch.long)

    # Move tensors to GPU
    test_input_sequences_tensor, test_output_words_tensor = (test_input_sequences_tensor.to('cuda'),
                                                             test_output_words_tensor.to('cuda'))

    # Create DataLoader for testing
    test_dataset = TensorDataset(test_input_sequences_tensor, test_output_words_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=False)

    # Call the prediction function
    predict_next_words(rnn_model, test_dataloader, vocab, max_loops=20)

if __name__ == "__main__":
    main()
