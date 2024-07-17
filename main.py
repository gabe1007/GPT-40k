import torch 
from tokenizer.bpe import BPE

def convert_to_tensor(text):
    """
    Converts a text to a tensor.

    Args:
        text (str): The text to be converted.

    Returns:
        torch.Tensor: A tensor representing the text.
    """
    bpe = BPE()
    encoded_data = bpe.encode(text)
    data = torch.tensor(encoded_data, dtype=torch.long)
    return data

def split(tensors):
    """
    Splits a list of tensors into training and testing sets.

    Parameters:
        tensors (list): A list of tensors to be split.

    Returns:
        tuple: A tuple containing two lists - the training set and the testing set.
    """
    limit = int(0.9 * len(tensors))
    train = tensors[:limit]
    test = tensors[limit:]

    return train, test


if __name__ == '__main__':
    with open('./Know_no_fear.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    text_ = convert_to_tensor(text[:1000])
    train, test = split(text_)
    print(test)
