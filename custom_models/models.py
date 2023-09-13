import torch
from torch import nn 
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class Model(nn.Module, cnn = None, dense = None, c=3):
    def __init__(self):
        super(Model, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),  # Alterei o tamanho do kernel para 3
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),  # Alterei o tamanho do kernel para 3
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Calculate the input size for the first linear layer
        self.embedding = nn.Flatten()
        in_features = self._get_cnn_output_size()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )

    def _get_cnn_output_size(self):
        # Dummy input to compute the size of the output of the CNN layers
        x = torch.randn(1, 3, 28, 28)
        cnn_output = self.CNN(x)
        return cnn_output.view(x.size(0), -1).size(1)

    def forward(self, inputs):
        cnn_output = self.CNN(inputs)
        embedding_output = self.embedding(cnn_output)  # Apply the flattening layer
        clf = self.classifier(embedding_output)  # Pass through the classifier
        return clf, embedding_output


def train_binary_clf(train_dl,model = Model(),criterion = torch.nn.BCELoss() ,optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)):

    nb_epochs = 100

    epoch_train_dict = {'Accuracy': [],
                'Loss': []}

    epoch_val_dict = {'Accuracy': [],
                'Loss': []}


    for epoch in range(nb_epochs):
        print(f'Epoch: {epoch}:')

        train_dict = {'Accuracy': [],
                'Loss': []}

        val_dict = {'Accuracy': [],
                    'Loss': []}

        model.train()
        for batch in train_dl:
            x, y = batch
            y_hat = model(x)
            optimizer.zero_grad()

            loss = criterion(y_hat, y.reshape(-1,1))
            training_loss_ = loss.item()
            loss.backward()
            optimizer.step()

            y_hat_ = y_hat > .5
            acc = accuracy_score(y.cpu().numpy(),y_hat_.cpu().numpy())
            train_dict['Accuracy'].append(acc)
            train_dict['Loss'].append(training_loss_)

        model.eval()
        for ix, batch in enumerate(val_dl):
            x, y = batch
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)

            val_acc_ = (y_hat.round() == y) / len(y)
            val_loss_ = loss.item()

            y_hat_ = y_hat > .5
            val_acc = accuracy_score(y.cpu().numpy(),y_hat_.cpu().numpy())

            val_dict['Accuracy'].append(val_acc)
            val_dict['Loss'].append(val_loss_)

        train_e_acc = np.mean(train_dict['Accuracy'])
        val_e_acc = np.mean(val_dict['Accuracy'])
        epoch_train_dict['Accuracy'].append(train_e_acc)
        epoch_val_dict['Accuracy'].append(val_e_acc)

        print(f'Accuracy: {train_e_acc} | Val Accuracy: {val_e_acc}')


        train_e_loss = np.mean(train_dict['Loss'])
        val_e_loss = np.mean(val_dict['Loss'])
        epoch_train_dict['Loss'].append(train_e_loss)
        epoch_val_dict['Loss'].append(val_e_loss)

        print(f'Loss: {train_e_loss} | Val Loss: {val_e_loss}')
    
    return epoch_train_dict,epoch_val_dict


