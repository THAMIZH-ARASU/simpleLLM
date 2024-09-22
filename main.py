import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

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
    text = """
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.
However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.
"My dear Mr. Bennet," said his lady to him one day, "have you heard that Netherfield Park is let at last?"
Mr. Bennet replied that he had not.
"But it is," returned she; "for Mrs. Long has just been here, and she told me all about it."
Mr. Bennet made no answer.
"Do not you want to know who has taken it?" cried his wife impatiently.
"You want to tell me, and I have no objection to hearing it."
This was invitation enough.
"Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and some of his servants are to be in the house by the end of next week."
"What is his name?"
"Bingley."
Is he married or single?"
"Oh! single, my dear, to be sure! A single man of large fortune; four or five thousand a year. What a fine thing for our girls!"
"How so? how can it affect them?"
"My dear Mr. Bennet," replied his wife, "how can you be so tiresome! You must know that I am thinking of his marrying one of them."
"Is that his design in settling here?"
"Design! nonsense, how can you talk so! But it is very likely that he may fall in love with one of them, and therefore you must visit him as soon as he comes."
"I see no occasion for that. You and the girls may go, or you may send them by themselves, which perhaps will be still better, for as you are as handsome as any of them, Mr. Bingley might like you the best of the party."
"My dear, you flatter me. I certainly have had my share of beauty, but I do not pretend to be anything extraordinary now. When a woman has five grown up daughters, she ought to give over thinking of her own beauty."
"In such cases, a woman has not often much beauty to think of."
"But, my dear, you must indeed go and see Mr. Bingley when he comes into the neighbourhood."
"It is more than I engage for, I assure you."
"But consider your daughters. Only think what an establishment it would be for one of them. Sir William and Lady Lucas are determined to go, merely on that account, for in general you know they visit no new comers. Indeed you must go, for it will be impossible for us to visit him if you do not."
"You are over-scrupulous, surely. I dare say Mr. Bingley will be very glad to see you; and I will send a few lines by you to assure him of my hearty consent to his marrying whichever he chooses of the girls; though I must throw in a good word for my little Lizzy."
"I desire you will do no such thing. Lizzy is not a bit better than the others; and I am sure she is not half so handsome as Jane, nor half so good-humoured as Lydia. But you are always giving her the preference."
"They have none of them much to recommend them," replied he; "they are all silly and ignorant like other girls; but Lizzy has something more of quickness than her sisters."
"Mr. Bennet, how can you abuse your own children in such a way? You take delight in vexing me. You have no compassion on my poor nerves."
"You mistake me, my dear. I have a high respect for your nerves. They are my old friends. I have heard you mention them with consideration these last twenty years at least."
"Ah, you do not know what I suffer."
"But I hope you will get over it, and live to see many young men of four thousand a year come into the neighbourhood."
"It will be no use to us, if twenty such should come, since you will not visit them."
"Depend upon it, my dear, that when there are twenty, I will visit them all."
Mr. Bennet was so odd a mixture of quick parts, sarcastic humour, reserve, and caprice, that the experience of three-and-twenty years had been insufficient to make his wife understand his character. Her mind was less difficult to develop. She was a woman of mean understanding, little information, and uncertain temper. When she was discontented, she fancied herself nervous. The business of her life was to get her daughters married; its solace was visiting and news.
"""



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
