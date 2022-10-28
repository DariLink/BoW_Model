import torch
import torch.nn as nn


class BagOfWordsClassifier(nn.Module):
    def __init__(self, vocab_size, number_hidden, number_labels):
        """
        defines network modules which will be used in this module
        Simple network an embedding and classification layer
        :param vocab_size: vocabulary size  e.g. number of most common elements
        number_hidden: number of hidden untis or dimensions to be included in embedding
        number_labels: number of output labels
        """
        super(BagOfWordsClassifier, self).__init__()
        self.number_hidden = number_hidden
        self.number_labels = number_labels
        self.embedding = nn.Embedding(vocab_size, number_hidden, padding_idx=0)
        self.input_layer = nn.Linear(number_hidden, number_hidden)
        self.hidden_layer = nn.Linear(number_hidden, number_hidden)
        self.output_layer = nn.Linear(number_hidden, self.number_labels)

    def forward(self, input) -> torch.FloatTensor:
        """
        A model forward pass.
        Calculates the sentence representation for the sentences in the given batch.
        The sentence word embeddings are pooled via mean pooling for a single fixed-sized representation,
        that is then fed to the classifier.
        The classifer is one input layer, a hidden layer and output layer with logits

        :param input: A 2-dimensional tensor that contains indices of the input words.
        :return: A 2-dimensional tensor with the 'logits' output of the classification layer.
        """
        bow_embedding = self.embedding(input)
        bow_embedding = bow_embedding.mean(dim=1)

        x = nn.functional.relu(self.input_layer(bow_embedding))
        x = nn.functional.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

    def predict(scores) -> torch.LongTensor:
        """
        converts the resulting logits into propabalities via softmax Function.
        """
        sm = torch.nn.Softmax(dim=1)
        probabilities = sm(scores)
        return probabilities
