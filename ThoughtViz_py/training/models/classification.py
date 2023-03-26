import torch
from torch import nn
from torch import optim




device = "cuda"
class ConvolutionalEncoder(nn.Module):
    def __init__(self, channels, observations, num_classes):
        super(ConvolutionalEncoder, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features = observations)
        self.conv1 = nn.Conv2d(observations, 32, kernel_size=(1, 4), stride=1,padding="same")
        self.conv2 = nn.Conv2d(32, 25, kernel_size=(channels, 1), stride=1,padding="same")
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=3)
        self.conv3 = nn.Conv2d(25, 50, kernel_size=(4, 25), stride=1,padding="same")
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=3)
        self.conv4 = nn.Conv2d(50, 100, kernel_size=(50, 2), stride=1,padding="same")
        self.bn2 = nn.BatchNorm1d(600)
        self.fc1 = nn.Linear(600, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.conv4(x)
        x= self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x
    



def convolutional_encoder_model(channels,observation,num_classes):
    model = ConvolutionalEncoder(channels,observation,num_classes)
    return model

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):

    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # print(X.shape)
        # print(y.shape)
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y.argmax(dim=1),
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    # val_loss, val_acc = 0, 0
    # with torch.inference_mode():
    #     for X,y in val_loader:
    #         X,y = X.to(device), y.to(device)
    #         y_pred = model(X)
    #         loss = loss_fn(y_pred,y)
    #         val_loss += loss
    #         val_acc += accuracy_fn(y_true=y,
    #                                y_pred=y_pred.argmax(dim=1))


def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    
    test_loss, test_acc = 0, 0 
    # Turn on inference context manager
    model.eval()
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y.argmax(dim=1),
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
            
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")

        return test_acc