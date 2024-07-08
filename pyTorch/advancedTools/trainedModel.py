import torch

def decode_sequence(indices, word_to_idx):
    """
    Decode the sequence of word indices into words using a provided word-to-index mapping.

    Args:
        indices (list): List of word indices.
        word_to_idx (dict): Mapping of words to their respective indices.

    Returns:
        list: List of words corresponding to the input indices.
    """
    return [word for idx in indices for word, word_idx in word_to_idx.items() if idx == word_idx]

def predict_next_words(model, test_dataloader, word_to_idx, max_loops):
    """
    Predict the next words using the provided model and dataloader.

    Args:
        model (torch.nn.Module): The trained model for prediction.
        test_dataloader (torch.utils.data.DataLoader): Dataloader for test data.
        word_to_idx (dict): Mapping of words to their respective indices.
        max_loops (int): Maximum number of loops for prediction.

    Returns:
        None
    """
    model.eval()

    for i, (input_sequence, target_word) in enumerate(test_dataloader):
        if i >= max_loops:
            user_response = input("Do you want to continue predicting? (yes/no): ")
            if user_response.lower() != 'yes':
                print("Prediction stopped.")
                break

        # Decode input sequence and target word
        input_sentence = decode_sequence(input_sequence[0], word_to_idx)
        target_word = decode_sequence([target_word[0]], word_to_idx)[0]

        # Prepare input sequence for prediction
        input_seq_tensor = input_sequence.to('cuda')

        with torch.no_grad():
            # Get model prediction
            output_probs = torch.softmax(model(input_seq_tensor), dim=1)
            _, predicted_index = torch.max(output_probs, 1)
            predicted_word = decode_sequence([predicted_index.item()], word_to_idx)[0]

        # Print results
        print(f"Sentence: {' '.join(input_sentence)}")
        print(f'Target: {target_word}')
        print(f'Prediction: {predicted_word}')
        print()

    print("Prediction finished.")
