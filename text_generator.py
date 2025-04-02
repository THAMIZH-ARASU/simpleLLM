"""Text generator using character-level LSTM.

This module implements a character-level text generator using PyTorch. It creates an LSTM-based neural
network that learns to predict the next character in a sequence, allowing it to generate text in a 
similar style to the training data.

Example:
    $ python text_generator.py
    Model saved to text_generator_model.pth
    Enter the text relevant to the input text: Once upon a time
    Generated text: Once upon a time there was a young man who...

Attributes:
    input_t (str): The input text used for training, imported from input_text.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from input_text import input_t

class TextDataset(Dataset):
    """Custom PyTorch Dataset for character-level text data.

    This dataset converts text to numerical indices and creates sequences of fixed length
    for training the model to predict the next character in a sequence.

    Args:
        text (str): The input text to be used for training.
        seq_length (int): Length of sequences to be generated.

    Attributes:
        char_to_idx (dict): Mapping from characters to indices.
        idx_to_char (dict): Mapping from indices to characters.
        vocab_size (int): Size of the vocabulary (number of unique characters).
        seq_length (int): Length of sequences to be generated.
        data (list): List of character indices representing the input text.
    """
    def __init__(self, text, seq_length):
        chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.seq_length = seq_length
        self.data = [self.char_to_idx[ch] for ch in text]
        
    def __len__(self):
        """Returns the total number of sequences in the dataset.

        Returns:
            int: Number of possible sequences to be extracted from text.
        """
        return len(self.data) - self.seq_length
        
    def __getitem__(self, idx):
        """Retrieves a single input/target sequence pair.

        Args:
            idx (int): Index of the sequence to retrieve.

        Returns:
            tuple: Contains:
                - torch.Tensor: Input sequence of character indices.
                - torch.Tensor: Target sequence of character indices (shifted by one position).
        """
        return (
            torch.tensor(self.data[idx:idx + self.seq_length]),
            torch.tensor(self.data[idx + 1:idx + self.seq_length + 1])
        )

class TextGeneratorModel(nn.Module):
    """LSTM-based model for character-level text generation.

    Args:
        vocab_size (int): Size of the vocabulary (number of unique characters).
        embed_size (int): Dimensionality of character embeddings.
        hidden_size (int): Number of features in the hidden state of the LSTM.

    Attributes:
        embedding (nn.Embedding): Embedding layer to convert character indices to vectors.
        lstm (nn.LSTM): LSTM layer for sequence modeling.
        fc (nn.Linear): Fully connected layer to output character probabilities.
    """
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(TextGeneratorModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length).
            hidden (tuple, optional): Hidden and cell states for the LSTM. Defaults to None.

        Returns:
            tuple: Contains:
                - torch.Tensor: Output logits of shape (batch_size, seq_length, vocab_size).
                - tuple: Updated hidden and cell states.
        """
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

def train_model(model, data_loader, num_epochs, device):
    """Train the text generation model.

    Args:
        model (TextGeneratorModel): The model to train.
        data_loader (DataLoader): DataLoader providing the training data.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to train the model on (CPU or CUDA).

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    model.train()  # Set the model to training mode
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, hidden = model(inputs)
            loss = criterion(outputs.view(-1, model.fc.out_features), targets.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(data_loader):.4f}")

def generate_text(model, start_text, gen_length, device, idx_to_char, char_to_idx):
    """Generate text using the trained model.

    Args:
        model (TextGeneratorModel): The trained model.
        start_text (str): Initial text to seed the generation process.
        gen_length (int): Number of characters to generate.
        device (torch.device): Device to run the model on (CPU or CUDA).
        idx_to_char (dict): Mapping from indices to characters.
        char_to_idx (dict): Mapping from characters to indices.

    Returns:
        str: Generated text starting with start_text.
    """
    model.eval()  # Set the model to evaluation mode
    input_seq = torch.tensor([char_to_idx[ch] for ch in start_text], dtype=torch.long).unsqueeze(0).to(device)
    generated_text = start_text
    
    with torch.no_grad():
        hidden = None
        for _ in range(gen_length):
            output, hidden = model(input_seq, hidden)
            output = output[:, -1, :]
            predicted_idx = torch.argmax(output, dim=1).item()
            generated_char = idx_to_char[predicted_idx]
            generated_text += generated_char
            input_seq = torch.cat((input_seq[:, 1:], torch.tensor([[predicted_idx]], dtype=torch.long).to(device)), dim=1)
            
    return generated_text

def main():
    """Main function to initialize, train, and run the text generator model."""
    # Initialize data
    text = input_t
    
    # Configure model and training parameters
    seq_length = 50
    dataset = TextDataset(text, seq_length)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    embed_size = 128
    hidden_size = 256
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create and train the model
    model = TextGeneratorModel(dataset.vocab_size, embed_size, hidden_size).to(device)
    train_model(model, data_loader, num_epochs, device)
    
    # Save the trained model
    model_path = "text_generator_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Load the model and generate text
    model.load_state_dict(torch.load(model_path))
    start_text = input("Enter the text relevant to the input text: ")
    generated_text = generate_text(model, start_text, 100, device, dataset.idx_to_char, dataset.char_to_idx)
    print(f"Generated text: {generated_text}")
    
if __name__ == "__main__":
    main()
