import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from input_text import input_t

class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.seq_length = seq_length
        self.data = [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx + self.seq_length]),
            torch.tensor(self.data[idx + 1:idx + self.seq_length + 1])
        )

class TextGeneratorModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(TextGeneratorModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

def train_model(model, data_loader, num_epochs, device):
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

if __name__ == "__main__":
    text = input_t
    
    seq_length = 50

    dataset = TextDataset(text, seq_length)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    embed_size = 128
    hidden_size = 256
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextGeneratorModel(dataset.vocab_size, embed_size, hidden_size).to(device)
    train_model(model, data_loader, num_epochs, device)

    model_path = "text_generator_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    model.load_state_dict(torch.load(model_path))
    start_text = "Mrs. Bennet was very"
    generated_text = generate_text(model, start_text, 100, device, dataset.idx_to_char, dataset.char_to_idx)
    print(f"Generated text: {generated_text}")
