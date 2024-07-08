import os
import re
from tqdm import tqdm
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# Download NLTK resources
import nltk
nltk.download('punkt')

sequence_length = 5  # Change between 3-10 for your desired sentence length BEST RESULTS when 10!!!

# Define special characters to be removed
SPECIAL_CHARACTERS = '!"#$%&\'’()*+,-./:;<=>?@[\\]^_`{|}~®'
# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def remove_numbers_and_special_chars(text):
    """
    Remove numbers and special characters from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The cleaned text.
    """
    cleaned_text = re.sub(r'[\d' + re.escape(string.punctuation) + ']', '', text)
    return cleaned_text


def remove_special_characters_and_lemmatize(text):
    """
    Remove special characters and lemmatize the text.

    Args:
        text (str): The input text.

    Returns:
        str: The cleaned and lemmatized text.
    """
    cleaned_text = remove_numbers_and_special_chars(text)
    tokens = word_tokenize(cleaned_text)
    lemmatized_text = ' '.join([lemmatizer.lemmatize(token) for token in tokens])
    return lemmatized_text


def get_current_path():
    """
    Get the current path of the script.

    Returns:
        str: The current path.
    """
    return os.path.dirname(os.path.abspath(__file__))


def create_directory(directory):
    """
    Create a directory if it doesn't exist.

    Args:
        directory (str): The directory path.

    Returns:
        None
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as e:
        print(f"Error creating directory: {e}")


def load_model_file(model_filename):
    """
    Load the model file.

    Args:
        model_filename (str): The name of the model file.

    Returns:
        str: The path to the model file.
    """
    script_dir = get_current_path()
    file_path = os.path.join(script_dir, 'base', model_filename)
    return file_path


def tokenize_and_build_vocab(text, model_file_name):
    """
    Tokenize the text and build vocabulary.

    Args:
        text (str): The input text.
        model_file_name (str): The name of the model file.

    Returns:
        tuple: A tuple containing input sequences and output words.
    """
    text = remove_special_characters_and_lemmatize(text)
    tokens = word_tokenize(text)
    vocab = {token: idx for idx, token in enumerate(set(tokens))}
    numerical_tokens = [vocab[token] for token in tokens]

    input_sequences = []
    output_words = []
    for i in tqdm(range(len(numerical_tokens) - sequence_length), desc="Processing Text"):
        input_sequence = numerical_tokens[i:i + sequence_length]
        output_word = numerical_tokens[i + sequence_length]
        input_sequences.append(input_sequence)
        output_words.append(output_word)

    model_file_name_without_extension = re.sub(r'\.(txt|csv)$', '', model_file_name)
    vocab_file_name = (f'vocab_{model_file_name_without_extension}'
                       f'_sequence_length_{sequence_length}.txt')
    vocab_file_path = os.path.join(get_current_path(), 'wordPairs', vocab_file_name)

    if not os.path.exists(vocab_file_path):
        with open(vocab_file_path, 'w', encoding='utf-8') as vocab_file:
            for token, idx in vocab.items():
                vocab_file.write(f"{token}: {idx}\n")

    return input_sequences, output_words


def write_word_pairs_to_file(word_pairs, output_file):
    """
    Write word pairs to a file.

    Args:
        word_pairs (list): List of word pairs.
        output_file (str): The output file path.

    Returns:
        None
    """
    with open(output_file, 'a', encoding='utf-8') as file:
        for input_sequence, output_word in word_pairs:
            file.write(f"Input Sequence: {input_sequence}, Output Word: {output_word}\n")


def load_text_data(file_path):
    """
    Load text data from a file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The text data.
    """
    sentences = file_to_sentence_list(file_path)
    return " ".join(sentences)


def file_to_sentence_list(file_path):
    """
    Convert a file to a list of sentences.

    Args:
        file_path (str): The path to the file.

    Returns:
        list: List of sentences.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    sentences = [sentence.strip() for sentence in re.split(
        r'(?<=[.!?])\s+', text) if sentence.strip()]

    return sentences


def preprocess_data(model_file_name):
    """
    Preprocess the data.

    Args:
        model_file_name (str): The name of the model file.

    Returns:
        tuple: A tuple containing paths to vocabulary file, train file, test file, and sequence length.
    """
    try:
        word_pairs_dir = os.path.join(get_current_path(), 'wordPairs')
        create_directory(word_pairs_dir)

        model_file_name_without_extension = re.sub(r'\.(txt|csv)$', '', model_file_name)
        output_file = os.path.join(word_pairs_dir, f'output_word_pairs_{model_file_name_without_extension}'
                                                   f'_sequence_length_{sequence_length}')

        output_file_base = os.path.join(word_pairs_dir, f'output_word_pairs_{model_file_name_without_extension}'
                                                        f''f'_sequence_length_{sequence_length}')
        output_file_exists = any(
            os.path.exists(f"{output_file_base}.txt") for file in os.listdir(word_pairs_dir)
        )

        if not output_file_exists:
            model_file_path = load_model_file(model_file_name)
            text_data = load_text_data(model_file_path)
            input_sequences, output_words = tokenize_and_build_vocab(text_data, model_file_name)
            word_pairs = list(zip(input_sequences, output_words))

            # Split the data into train and test sets
            train_word_pairs, test_word_pairs = train_test_split(word_pairs, test_size=0.1, random_state=42)

            # Write word pairs to the output files
            write_word_pairs_to_file(train_word_pairs, f"{output_file}_train.txt")
            write_word_pairs_to_file(test_word_pairs, f"{output_file}_test.txt")

        vocab_file_path = os.path.join(
            get_current_path(), 'wordPairs',
            f'vocab_{model_file_name_without_extension}_sequence_length_{sequence_length}.txt'
        )

        train_word_pairs_file_path = f"{output_file}_train.txt"
        test_word_pairs_file_path = f"{output_file}_test.txt"

        return vocab_file_path, train_word_pairs_file_path, test_word_pairs_file_path, sequence_length

   

    except Exception as e:
        print(f"Error preprocessing data: {e}")