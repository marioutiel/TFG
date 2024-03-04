import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

def binary_acc(y_pred, y_test):
    correct_results_sum = (torch.round(y_pred) == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

class GamesData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return len(self.X)

class BinaryFCNN(nn.Module):
    def __init__(self, input_size):
        super(BinaryFCNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        
    def forward(self, inputs):
        x = self.leaky_relu(self.fc1(inputs))
        x = self.bn1(x)
        
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.bn2(x)
        
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        
        return x

class BinaryFCNN2(nn.Module):
    def __init__(self, input_size):
        super(BinaryFCNN2, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)
        
    def forward(self, inputs):
        x = self.leaky_relu(self.fc1(inputs))
        x = self.bn1(x)
        
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.bn2(x)
        
        x = self.dropout(x)
        x = self.leaky_relu(self.fc3(x))
        x = self.bn3(x)
        
        x = self.dropout(x)
        x = self.leaky_relu(self.fc4(x))
        x = self.bn4(x)
        
        x = self.dropout(x)
        x = self.sigmoid(self.fc5(x))
        
        return x

class BinaryFCNN3(nn.Module):
    def __init__(self, input_size):
        super(BinaryFCNN3, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, inputs):
        x = self.leaky_relu(self.fc1(inputs))
        
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        
        return x

class BinaryFCNN4(nn.Module):
    def __init__(self, input_size):
        super(BinaryFCNN4, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, inputs):
        x = self.leaky_relu(self.fc1(inputs))
        
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        
        x = self.dropout(x)
        x = self.leaky_relu(self.fc3(x))
        
        x = self.dropout(x)
        x = self.leaky_relu(self.fc4(x))
        
        x = self.dropout(x)
        x = self.sigmoid(self.fc5(x))
        
        return x
    
class SimpleRNN(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(SimpleRNN, self).__init__()

        self.RNN = nn.RNN(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, inputs):
        x = self.layer_norm(inputs)
        x = self.gelu(x)
        
        x, _ = self.RNN(x)
        x = self.dropout(x)
        
        return x
    
class BinaryRNN(nn.Module):
    
    def __init__(self, n_rnn_layers, input_size, rnn_dim, dropout=0.5):
        super(BinaryRNN, self).__init__()
        
        self.fc = nn.Linear(input_size, rnn_dim)
        
        self.recursive_layers = nn.Sequential(*[
            SimpleRNN(rnn_dim=rnn_dim, hidden_size=rnn_dim,
                      dropout=dropout, batch_first= i==0)
            for i in range(n_rnn_layers)
        ])
        
        self.fc1 = nn.Linear(rnn_dim, input_size)
        self.batchnorm = nn.BatchNorm1d(input_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        x = self.fc(inputs)
        x = self.recursive_layers(x)
        
        x = self.batchnorm(self.fc1(x))
        x = self.gelu(x)
        
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        
        return x
    
class BinaryLSTM(nn.Module):
    
    def __init__(self, input_size):
        super(BinaryLSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, input_size, batch_first=True)
        
        self.fc1 = nn.Linear(input_size, 6)
        self.fc2 = nn.Linear(6, 1)
        
        self.leaky = nn.LeakyReLU()
        
        self.batchnorm = nn.BatchNorm1d(6)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        x, _ = self.lstm1(inputs)
        
        x = self.batchnorm(self.fc1(x))
        x = self.leaky(x)
        
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        
        return x