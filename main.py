import torch
from torch.utils.data import DataLoader
import pandas as pd
from nlpds.data import Dataset, custom_collate_fn
from nlpds.model import BagOfWordsClassifier
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


if __name__ == '__main__':

    # create pd dataframes to pass to Dataset class
    colnames = ['label', 'text']
    df_train = pd.read_csv('train.tsv', sep='\t', names=colnames, header=None)
    df_test = pd.read_csv('test.tsv', sep='\t', names=colnames, header=None)
    df_dev = pd.read_csv('dev.tsv', sep='\t', names=colnames, header=None)

    vocab_size = 5000

    # bonus
    '''
    dict_new_labels = {0:0, 1:0, 2:1, 3:2, 4:2}
    df_train['label'] = df_train['label'].apply(lambda x: dict_new_labels[x])
    df_test['label'] = df_test['label'].apply(lambda x: dict_new_labels[x])
    df_dev['label'] = df_dev['label'].apply(lambda x: dict_new_labels[x])
    '''

    train_dataset = Dataset(df_train['text'].values.tolist(), df_train['label'].values.tolist(), vocab_size)
    dev_dataset = Dataset(df_dev['text'].values.tolist(), df_dev['label'].values.tolist(), vocab_size)
    test_dataset = Dataset(df_test['text'].values.tolist(), df_test['label'].values.tolist(), vocab_size)

    number_hidden = 500
    number_labels = 5 # bonus exercise 1 can be implemented here in main without adjusting modules
    model = BagOfWordsClassifier(vocab_size, number_hidden, number_labels)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    batch_size = 3
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    tb = SummaryWriter()
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        for x_batch, y_batch in DataLoader(train_dataset, batch_size=batch_size,
                                           collate_fn=custom_collate_fn):

            #clear gradients
            model.zero_grad()
            # forward propagation
            scores = model(x_batch)
            # loss function
            loss = criterion(scores, y_batch.type(torch.LongTensor))
            # zero previous gradients
            optimizer.zero_grad()
            # back-propagation
            loss.backward()
            # gradient descent or adam step
            optimizer.step()

                                    ##### validation part #####
        valid_loss = 0.0
        num_correct = 0
        model.eval()
        for x_batch, y_batch in DataLoader(dev_dataset, batch_size=batch_size,
                                           collate_fn=custom_collate_fn):
            # Forward Pass
            scores = model(x_batch)
            # Find the Loss
            loss = criterion(scores, y_batch.type(torch.LongTensor))
            valid_loss += loss.item()
            predictions = scores.argmax(dim=-1)
            #print(predictions)
            num_correct += (predictions == y_batch).float().sum()

        tb.add_scalar("Loss_val", valid_loss/len(dev_dataset), epoch)
        tb.add_scalar("Accuracy_val", num_correct / len(dev_dataset) * 100, epoch)

                                    ##### test part #####
        model.eval()

        test_loss = 0.0
        num_correct = 0
        with torch.no_grad():
            for x_batch, y_batch in DataLoader(test_dataset, batch_size=batch_size,
                                               collate_fn=custom_collate_fn):

                scores = model(x_batch)

                y_batch = y_batch.type(torch.LongTensor)
                loss = criterion(scores, y_batch)
                test_loss += loss.item()
                predictions = scores.argmax(dim=-1)
                num_correct += (predictions == y_batch).float().sum()
                probabilities = BagOfWordsClassifier.predict(scores)

                # show predictions for the labels
                for ix_text, i in enumerate(probabilities):
                    for ix_label, j in enumerate(i):
                        print(f"Item: {str(ix_text)} Probability for {str(ix_label)} label is %5.2f " % (j.item()*100))

            tb.add_scalar("Loss_test", test_loss/len(test_dataset), epoch)
            tb.add_scalar("Accuracy_test", num_correct / len(test_dataset) * 100, epoch)

writer.flush()
tb.close()
