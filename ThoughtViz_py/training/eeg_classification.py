import os
import pickle

import torch
from torch import nn
from torch import optim
from torch.utils import data as data_utils

from models.classification import *


import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  # Note: you need the "raw" GitHub URL for this to work
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import accuracy_fn


class EEG_Classifier():

    def __init__(self,num_classes,dataset):
        self.num_classes = num_classes
        self.dataset = dataset
        self.eeg_pkl_file = os.path.join('../data/eeg/', self.dataset, 'data.pkl')

    def train(self, model_save_dir, run_id, batch_size, num_epochs):

        data = pickle.load(open(self.eeg_pkl_file, 'rb'), encoding='bytes')

        x_train, y_train, x_test, y_test = data[b'x_train'], data[b'y_train'], data[b'x_test'], data[b'y_test']

        x_train = torch.tensor(x_train, dtype=torch.float32).view(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32).view(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])
        y_test = torch.tensor(y_test, dtype=torch.float32)

        print(x_train.shape)
        classifier = ConvolutionalEncoder(x_train.shape[2],x_train.shape[1], self.num_classes)

        classifier = classifier.to(device)

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        saved_model_file = os.path.join(model_save_dir, str(run_id) + '_final' + '.pt')

        filepath = os.path.join(model_save_dir, str(run_id) + "-model-improvement-{epoch:02d}-{val_accuracy:.2f}.pt")

        # callback_checkpoint = nn.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=False, save_best_only=True, mode='max')

        optimizer = optim.SGD(classifier.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-6, nesterov=True)

        criterion = nn.CrossEntropyLoss()

        train_dataset = data_utils.TensorDataset(x_train, y_train)

        test_dataset = data_utils.TensorDataset(x_test, y_test)

        test_loader = data_utils.DataLoader(test_dataset, shuffle=True)

        train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


        for epoch in range(num_epochs):

            print(f"Epoch: {epoch}\n---------")

            train_step(classifier, train_loader, criterion, optimizer, accuracy_fn, device)

            # accuracy = test_step(test_loader, classifier, criterion, accuracy_fn, device)

                
        torch.save(classifier.state_dict(), saved_model_file)
        accuracy = test_step(test_loader, classifier, criterion, accuracy_fn, device)    
        
        return accuracy
    

if __name__ == '__main__':

    batch_size, num_epochs = 128, 100
    # digit_classifier = EEG_Classifier(10, 'digit')
    # acc1 = digit_classifier.train('./eeg_digit_classification', 1, batch_size, num_epochs)

    # print("Accuracy of digit classification: ", acc1)

    char_classifier = EEG_Classifier(10, 'char')
    acc2 = char_classifier.train('./eeg_char_classification', 1, batch_size, num_epochs)

    print("Accuracy of char classification: ", acc2)

    image_classifier = EEG_Classifier(10, 'image')
    acc3 = image_classifier.train('./eeg_image_classification', 1, batch_size, num_epochs)

    print("Accuracy of image classification: ", acc3)





