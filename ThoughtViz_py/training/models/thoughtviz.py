import torch
import torch.nn as nn
from layers.mog_layer import MoGLayer
import torch.nn.functional as F


class DiscriminatorRGB(nn.Module):

    def __init__(self):
        super(DiscriminatorRGB, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.BatchNorm2d(16)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.BatchNorm2d(32)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.BatchNorm2d(64)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.BatchNorm2d(128)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.BatchNorm2d(256)
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.BatchNorm2d(512)
        )


        self.fc = nn.Linear(4608, 1)
        self.sig = nn.Sigmoid()


    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        x = self.sig(x)



        return x
    
def calcPad(kernel_size,input_size,stride):
    padding_h = ((stride - 1) * input_size[0] - stride + kernel_size) // 2
    padding_w = ((stride - 1) * input_size[1] - stride + kernel_size) //2

    return padding_h,padding_w
    

class GeneratorRGB(nn.Module):
    def __init__(self, noise_dim, feature_dim):
        super(GeneratorRGB, self).__init__()
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim
        self.mog_layer = MoGLayer(noise_dim, (-0.2,0.2),(-1,1))
        self.dense1 = nn.Linear(noise_dim, feature_dim)
        self.bn1 = nn.BatchNorm1d(feature_dim,momentum=0.8)
        self.dense2 = nn.Linear(feature_dim, 512*4*4)
        self.bn2 = nn.BatchNorm2d(512,momentum=0.8)
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2,output_padding=1)
        self.bn3 = nn.BatchNorm2d(256,momentum=0.8)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2,output_padding=1)
        self.bn4 = nn.BatchNorm2d(128,momentum=0.8)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2,output_padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2,output_padding=1)

        # self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(256,momentum=0.8)
        # self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)
        # self.bn4 = nn.BatchNorm2d(128,momentum=0.8)
        # self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        # self.bn5 = nn.BatchNorm2d(64)
        # self.conv4 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2)
        
    def forward(self, noise, eeg):
        x = self.mog_layer(noise)
        x = self.dense1(x)
        x = F.tanh(x)
        x = x*eeg
        x = self.bn1(x)
        x = self.dense2(x)
        x = F.relu(x)
        x = x.view(x.size(0), 512, 4, 4)
        x = self.bn2(x)
        x = F.relu(self.conv1(x))
        x = self.bn3(x)
        x = F.relu(self.conv2(x))
        x = self.bn4(x)
        x = F.relu(self.conv3(x))
        x = self.bn5(x)
        x = F.relu(self.conv4(x))
        x = F.tanh(x)

        return x
    
class CombinedDisClassifier(nn.Module):
    def __init__(self, d_model, c_model):
        super(CombinedDisClassifier, self).__init__()
        self.discriminator = d_model
        self.classifier = c_model

        for p in self.classifier.parameters():
            p.requires_grad = False
        
    def forward(self, img):
        fake = self.discriminator(img)
        aux = self.classifier(img)
        return fake, aux


class CombinedGD(nn.Module):
    def __init__(self, g_model, d_model):
        super(CombinedGD, self).__init__()
        self.generator = g_model
        self.discriminator = d_model
        
    def forward(self, noise, eeg):
        generated_img = self.generator(noise, eeg)
        fake,aux = self.discriminator(generated_img)
        return fake,aux

        


    

