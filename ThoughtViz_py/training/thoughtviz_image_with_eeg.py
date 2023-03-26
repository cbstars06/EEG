from utils.data_input_util import *
from utils.image_utils import *
import logging
import os
from random import randint
from torchinfo import summary
import numpy as np

import random
from PIL import Image
import pickle
import torch
import torchvision
import torch.distributed as dist
import torch.nn as nn
from torch.utils import data as data_utils
from helper_functions import accuracy_fn

from models.thoughtviz import DiscriminatorRGB, GeneratorRGB, CombinedDisClassifier,CombinedGD
from models.classification import *

device = "cuda"

# def train_step2(model: torch.nn.Module,
#                data_loader: torch.utils.data.DataLoader,
#                loss_fn: torch.nn.Module,
#                optimizer: torch.optim.Optimizer,
#                accuracy_fn,
#                device: torch.device = device):

#     train_loss, train_acc = 0, 0
#     for batch, (X, y) in enumerate(data_loader):
#         # Send data to GPU
#         X, y = X.to(device), y.to(device)

#         # print(X.shape)
#         # print(y.shape)
#         # 1. Forward pass
#         fake,aux = model(X)

#         # 2. Calculate loss
#         loss = loss_fn(fake, aux)
#         train_loss += loss
#         train_acc += accuracy_fn(y_true=aux.argmax(dim=1),
#                                  y_pred=fake.argmax(dim=1)) # Go from logits -> pred labels

#         # 3. Optimizer zero grad
#         optimizer.zero_grad()

#         # 4. Loss backward
#         loss.backward()

#         # 5. Optimizer step
#         optimizer.step()

#     # Calculate loss and accuracy per epoch and print out what's happening
#     train_loss /= len(data_loader)
#     train_acc /= len(data_loader)
#     print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%") 


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# def train_step(model: torch.nn.Module,
#                data_loader: torch.utils.data.DataLoader,
#                loss_fn: torch.nn.Module,
#                optimizer: torch.optim.Optimizer,
#                accuracy_fn,
#                batch_size,
#                eeg_classfier,
#                y_test,
#                generator,
#                input_noise_dim=100,
#                device: torch.device = device):

#     for batch,(X,y) in enumerate(data_loader):
#         noise = np.random.uniform(-1, 1, (batch_size, input_noise_dim))

#         random_labels = np.random.randint(0, 10, batch_size)

#         one_hot_vectors = [to_categorical(label, 10) for label in random_labels]

#         eeg_feature_vectors = np.array([eeg_classfier[random.choice(np.where(y_test == random_label)[0])] for random_label in random_labels])

#         real_images = X
#         real_labels = y

#         generated_images = generator(noise,eeg_feature_vectors)


def train_on_batch(model:torch.nn.Module,
                   X,y,
                   optim,loss_fn):
    

    X,y = X.to(device),y.to(device)
    y = y.unsqueeze(1)
    # print(X.shape)
    # print(y.shape)

    y_pred,labels = model(X)

    loss = loss_fn(y_pred,y)

    train_loss = loss

    optim.zero_grad()

    loss.backward()

    optim.step()

    return train_loss

def train_on_batch2(d:torch.nn.Module,
                    g:torch.nn.Module,
                    X,y,
                    optim,loss_fn1,loss_fn2):
    
    X = [X[0].to(device),X[1].to(device)]
    y = [y[0].view(100,1).to(device).to(torch.float32),y[1].to(device).type(torch.float32)]

    gen_imgs = g(X[0],X[1])

    y_pred,labels = d(gen_imgs)

    loss_bce = loss_fn1(y_pred,y[0])
    loss_ce = loss_fn2(labels,y[1])

    loss = loss_bce + loss_ce

    optim.zero_grad()

    loss.backward()

    # gen_imgs.grad = None
    # gen_imgs.retain_grad()

    # print(loss.grad)
    # gen_imgs.backward(loss.grad)

    optim.step()

    train_loss = loss

    return train_loss


def train_gan(input_noise_dim, batch_size, epochs, data_dir, saved_classifier_model_file, model_save_dir, output_dir, classifier_model):

    imagenet_folder = "./images/ImageNet-Filtered"
    num_classes = 10

    feature_encoding_dim = 100

    d_adam_lr = 0.00005
    d_adam_beta_1 = 0.5

    g_adam_lr = 0.00003
    g_adam_beta_1 = 0.5

    # load data and compile discriminator, generator models depending on the dataaset
    x_train, y_train, x_test, y_test = load_image_data(imagenet_folder, patch_size=(64, 64))

    x_train = torch.tensor(x_train, dtype=torch.float32).view(x_train.shape[0], 3, x_train.shape[1], x_train.shape[2])
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32).view(x_test.shape[0], 3, x_test.shape[1], x_test.shape[2])
    y_test = torch.tensor(y_test, dtype=torch.float32)
    print("Loaded Images Dataset.", )

    train_dataset = data_utils.TensorDataset(x_train, y_train)

    test_dataset = data_utils.TensorDataset(x_test, y_test)

    test_loader = data_utils.DataLoader(test_dataset, shuffle=True)

    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    c = classifier_model.to(device)

    # c.load_state_dict(torch.load(os.path.join("./image_classifier","vgg_final4.pt")))

    # summary(c, (1,3, 64, 64),
    #     col_names=["input_size", "output_size", "num_params", "trainable"],
    #     col_width=20,
    #     row_settings=["var_names"]
    # )

    # c.load_state_dict(torch.load(os.path.join("./image_classifier","vgg_final4.pt")))

    # for epoch in range(2000):
    #     print(f"Epoch: {epoch}\n---------")
    #     train_step(c,train_loader,torch.nn.CrossEntropyLoss(),optimizer = torch.optim.SGD(c.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-6),accuracy_fn=accuracy_fn,device=device)

    # torch.save(c.state_dict(), os.path.join("./image_classifier","vgg_final5.pt"))

    # model_save_dir = "./discriminator"

    # saved_model_file = os.path.join(model_save_dir, "dis_final1.pt")

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    d = DiscriminatorRGB()

    # d.load_state_dict(torch.load(os.path.join(model_save_dir, 'd_' + '350' + '.pth')))

    d2 = CombinedDisClassifier(d, c).to(device)

    summary(d2, (1,3, 64, 64),col_names=["input_size", "output_size", "num_params", "trainable"],col_width=20,row_settings=["var_names"])



    optim_d = torch.optim.Adam(d2.parameters(), lr=d_adam_lr, betas=(d_adam_beta_1, 0.999))

    loss_d = torch.nn.BCELoss()







    # # summary(d2, (1,3, 64, 64),col_names=["input_size", "output_size", "num_params", "trainable"],
    # #     col_width=20,
    # #     row_settings=["var_names"])

    # # torch.save(d2.state_dict(), saved_model_file)

    g = GeneratorRGB(input_noise_dim, feature_encoding_dim).to(device)



    # g.load_state_dict(torch.load(os.path.join(model_save_dir, 'g_' + '350' + '.pth')))

    summary(g, [(1,100),(1,100)],col_names=["input_size", "output_size", "num_params", "trainable"],col_width=20,row_settings=["var_names"])

    # d_on_g = CombinedGD(g, d)

    optim_g = torch.optim.Adam(g.parameters(), lr=g_adam_lr, betas=(g_adam_beta_1, 0.999))

    loss_g2 = torch.nn.CrossEntropyLoss()

    loss_g1 = torch.nn.BCELoss()

    # eeg_data = pickle.load(open(os.path.join(data_dir, 'data.pkl'), "rb"))

    # classifier = torch.load(saved_classifier_model_file)

    eeg_classifier = ConvolutionalEncoder(14,1,10).to(device)

    eeg_classifier.fc1.register_forward_hook(get_activation('fc1'))

    eeg_classifier.load_state_dict(torch.load(saved_classifier_model_file,map_location=torch.device(device)))

    eeg_data = pickle.load(open(os.path.join(data_dir,"data.pkl"), 'rb'),encoding='bytes')

    x_test = eeg_data[b'x_test']
    x_test = torch.tensor(x_test, dtype=torch.float32).view(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]).to(device)
    y_test = eeg_data[b'y_test']

    y_test = np.array([np.argmax(y) for y in y_test])

    output = eeg_classifier(x_test)

    for epoch in range(epochs):
        print("Epoch is ", epoch)

        print("Number of batches", int(x_train.shape[0]/batch_size))
        for batch,(X,y) in enumerate(train_loader):
            noise = np.random.uniform(-1, 1, (batch_size, input_noise_dim))
            

            random_labels = np.random.randint(0, 10, batch_size)


            one_hot_vectors = [to_categorical(label, 10) for label in random_labels]

            # print(activation['fc1'][0].shape)
            eeg_feature_vectors = np.array([activation['fc1'].cpu().detach().numpy()[random.choice(np.where(y_test == random_label)[0])] for random_label in random_labels])
            eeg_feature_vectors = torch.tensor(eeg_feature_vectors[0], dtype=torch.float32).to(device)

            noise = torch.tensor(noise, dtype=torch.float32).to(device)
            # print(noise.shape)
            # print(eeg_feature_vectors.shape)
            random_labels = torch.tensor(random_labels, dtype=torch.float32).to(device)
            real_images = X
            real_labels = y

            generated_images = g(noise,eeg_feature_vectors)

            d_loss_real = train_on_batch(d2,X,torch.ones(X.shape[0]),optim_d,loss_d)    
            d_loss_fake = train_on_batch(d2,generated_images.detach(),torch.zeros(batch_size),optim_d,loss_d)

            d_loss = (d_loss_real + d_loss_fake)*0.5

            for param in d.parameters():
                param.requires_grad = False

            g_loss = train_on_batch2(d2,g,[noise,eeg_feature_vectors],[torch.ones(batch_size),torch.from_numpy(np.array(one_hot_vectors).reshape(batch_size, num_classes))],optim_g,loss_g1,loss_g2)

            for param in d.parameters():
                param.requires_grad = True

    
        if epoch % 100 == 0:
            image = combine_rgb_images(generated_images)
            image = image * 255.0
            img_save_path = os.path.join(output_dir, str(epoch) + "_g" + ".png")
            print("Saving image to ", img_save_path)
            Image.fromarray(image.astype(np.uint8)).save(img_save_path)

        if epoch % 100 == 0:
            test_image_count = 50000
            test_noise = np.random.uniform(-1, 1, (test_image_count, input_noise_dim))
            test_noise = torch.tensor(test_noise,dtype=torch.float32)
            test_labels = np.random.randint(0, 10, test_image_count)
            print("Hii")
            eeg_feature_vectors_test = np.array([activation['fc1'].cpu().detach().numpy()[random.choice(np.where(y_test == test_label)[0])] for test_label in test_labels])
            eeg_feature_vectors_test = torch.tensor(eeg_feature_vectors_test, dtype=torch.float32).to(device)
            test_labels = torch.tensor(test_labels, dtype=torch.float32).to(device)
            test_images = g(test_noise, eeg_feature_vectors_test)
            test_images = test_images * 255.0
            # inception_score = get_inception_score([test_image for test_image in test_images], splits=10)

        print("Epoch %d d_loss : %f" % (epoch, d_loss))
        print("Epoch %d g_loss : %f" % (epoch, g_loss.item()))
        # print("Epoch %d inception_score : %f" % (epoch, inception_score[0]))
        if epoch % 50 == 0:
            # save generator and discriminator models along with the weights
            torch.save(g.state_dict(), os.path.join(model_save_dir, 'g_' + str(epoch) + '.pth'))
            torch.save(d.state_dict(), os.path.join(model_save_dir, 'd_' + str(epoch) + '.pth'))



def train():
    dataset = 'Image'
    batch_size = 100
    run_id = 1
    epochs = 10000
    model_save_dir = os.path.join('./saved_models/thoughtviz_image_with_eeg/', dataset, 'run_' + str(run_id))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    output_dir = os.path.join('./outputs/thoughtviz_image_with_eeg/', dataset, 'run_' + str(run_id))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
 
    classifier_model = torchvision.models.vgg16().to(device)
    # classifier_model.load_state_dict(torch.load('./image_classifier/vgg16_bn.pth'))

    num_features = classifier_model.classifier[6].in_features
    classifier_model.classifier[6] = nn.Sequential(
       nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.4),
       nn.Linear(256, 10), nn.LogSoftmax(dim=1))

    classifier_model.load_state_dict(torch.load('./image_classifier/vgg_final4.pt', map_location=torch.device(device)))

    for param in classifier_model.features.parameters():
        param.requires_grad = False
    # classifier_model.classifier = torch.nn.Sequential(
    #     torch.nn.Dropout(p=0.2, inplace=True), 
    #     torch.nn.Linear(in_features=1280, 
    #                     out_features=10, # same number of output units as our number of classes
    #                     bias=True)).to(device)


    eeg_data_dir = os.path.join('../data/eeg/', dataset.lower())
    eeg_classifier_model_file = os.path.join('./eeg_image_classification', '1_final.pt')

    train_gan(input_noise_dim=100, batch_size=batch_size, epochs=epochs,data_dir=eeg_data_dir, saved_classifier_model_file=eeg_classifier_model_file, model_save_dir=model_save_dir, output_dir=output_dir, classifier_model=classifier_model)





if __name__ == '__main__':
    # dist.init_process_group(backend='gloo')
    train()