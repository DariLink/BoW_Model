from torch.utils.data.dataset import Dataset as TorchDataset
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
lemmatizer = WordNetLemmatizer()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_collate_fn(batch):
    """
    This function is used to pad the elements in the batch with 0 as unknown token.
    """

    xx, yy = zip(*batch)
    xx_pad = pad_sequence(list(xx), batch_first=True, padding_value=0)
    yy = torch.stack(list(yy), dim=0)
    xx_pad = torch.squeeze(xx_pad, 2)

    return xx_pad,  yy


def text_preprocess(text):
    """
    This Function is used to preprocess the given text with stop words removal und lemmatizing
    :param text: text to preprocess
    :return: preprocessed text
    """
    stop_words = set(stopwords.words('english'))
    # lowercase and add lemmatized token to list if not stopwords
    return_text = []
    list_remove = [',', '(', ')', '\n', '"', '--', '"', '...']
    for i in text.split():
        if i not in stop_words and i not in list_remove: return_text.append(lemmatizer.lemmatize(i.lower()))
    return return_text


def get_word_to_id(vocab_size, text):
    """
    get a dict with words and their corresponding ids
    :param vocab_size: number of most common words
    :param text: list of text
    :return: word_to_id dict
    """
    # tokenize text
    text_preprocessed = [text_preprocess(i) for i in text]

    # build dictionary
    # get all tokens in one list
    all_tokens = [token for sen in text_preprocessed for token in sen]

    # populate dict, take only no of vocab_size common words
    word_to_id = {
        x[0]: i
        for i, x in enumerate(Counter(all_tokens).most_common()[: (vocab_size - 2)])
    }
    # word at index 0 is ".", should be removed anyway, 0 ist reserved for padding
    word_to_id.update({'<pad>': 0})
    return word_to_id


class Dataset(TorchDataset):
    def __init__(self, text, labels, vocab_size):
        """
        builds from text data vocabulary of words and assigns every word its unique id in the vocab dict

        :param text: list of texts (tweets)
        :param labels: list of corresponding labels
        :param vocab_size: size of vocab
        """
        super(Dataset, self).__init__()
        self.labels = labels
        self.text = text
        self.vocab_size = vocab_size
        self.word_to_id = get_word_to_id(vocab_size, text)


    def __len__(self):
        """
        Returns the length of this dataset.
        :return: The length of this dataset.
        """
        return len(self.text)

    def __getitem__(self, item):
        """
        Returns the item (the features and corresponding label) at the given index. vocab_size - 1 ist reserved for
        unknown words

        :param item: The index of the item to get.
        :return: The item at the corresponding index. This is the label and features encoded as word_ids
        """

        feats = torch.LongTensor(
            [self.word_to_id.get(i, self.vocab_size-1) for i in text_preprocess(self.text[item])]
        )
        label = torch.tensor(self.labels[item], dtype=torch.int8)
        feats = feats.unsqueeze(1)
        return feats, label
