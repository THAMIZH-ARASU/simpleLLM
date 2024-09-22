# PyTorch Text Generation Model

This project implements a character-level text generation model using PyTorch. The model is trained on text data and can generate new sequences of text based on a seed input.

## Features

- **Custom Dataset**: The project defines a dataset class that converts input text into sequences of characters, which are then used for training the model.
- **LSTM-based Model**: The model uses an LSTM (Long Short-Term Memory) architecture to learn from the input sequences and generate text.
- **Text Generation**: After training, the model can generate new text by predicting character sequences from a starting seed.
- **GPU Support**: Utilizes GPU if available for faster training.

## Dataset

The dataset used for training the model is a portion of the text from *Pride and Prejudice* by Jane Austen. The text is converted into sequences of characters for training.

## Requirements

To run this project, you need to install the following dependencies:

- `torch`: For building and training the model.
- `numpy`: For array manipulations.

You can install the dependencies using `pip`:

```bash
pip install torch numpy


## Code Structure
- **TextDataset**: A PyTorch dataset that prepares the input text for training by converting it into sequences of character indices.
- **TextGeneratorModel**: The LSTM-based model that takes input sequences and generates character predictions.
- **Training Function**: A function that trains the model using cross-entropy loss and Adam optimizer.
- **Text Generation Function**: A function that generates text from a trained model by feeding it a starting sequence.
- **Main Script**: A script to load data, train the model, save the trained model, and generate text.

## Model Architecture
The model consists of:

**Embedding Layer**: Converts input character indices into dense vectors.
**LSTM Layer**: Learns temporal dependencies from the input sequences.
**Fully Connected Layer**: Outputs probabilities for the next character in the sequence.

## Usage

To run this project, clone the repository and run the following command,

```bash
python main.py

License
This project is licensed under the GNU License.
