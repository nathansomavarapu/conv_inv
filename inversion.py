import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models


class InverseNet(nn.Module):

    def __init__(self, l):
        super(InverseNet, self).__init__()
        self.an = models.alexnet(pretrained=True)
        feats = self.an.features
        cl = self.an.classifier
        if l < 5:
            self.conv_lin = 'conv'
            split_point = 0
            conv_cntr = 0
            for lay in feats:
                if isinstance(lay, nn.Conv2d):
                    conv_cntr += 1
                split_point += 1

                if conv_cntr == l:
                    break
                
            self.an.features = nn.Sequential(*list(feats)[:split_point+1])
                
        elif l >=6 and l <=8:
            self.conv_lin = 'lin'
            split_point = 0
            lin_cntr = 5
            for lay in feats:
                if isinstance(lay, nn.Linear):
                    lin_cntr += 1
                split_point += 1

                if lin_cntr == l:
                    break
                
            self.an.cl = nn.Sequential(*list(cl)[:split_point+1])

        # Freeze base network parameters
        for param in self.an.parameters():
            param.requires_grad = False

        self.lin_net1 = nn.Sequential(nn.Linear(1000, 4096),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(4096, 4096),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(4096, 4096),
                                     nn.LeakyReLU(0.2, inplace=True))

        self.lin_net2 = nn.Sequential(nn.ConvTranspose2d(256, 256, 5, stride=2),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.ConvTranspose2d(256, 128, 5, stride=2),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.ConvTranspose2d(128, 64, 5, stride=2),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.ConvTranspose2d(64, 32, 5, stride=2),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      nn.ConvTranspose2d(32, 3, 8, stride=2),
                                      nn.LeakyReLU(0.2, inplace=True))
    
    def forward(self, x):
        x = self.an(x)
        if self.conv_lin == 'conv':
            pass
        elif self.conv_lin == 'lin':
            x = self.lin_net1(x)
            x = x.view(x.size(0), 256, 4, 4)
            x = self.lin_net2(x)
        return x

def main():
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    epochs = 1
    batch_size = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    imagenet_train = torchvision.datasets.ImageFolder('data/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(imagenet_train, batch_size=batch_size, shuffle=True, num_workers=2)

    imagenet_test = torchvision.datasets.ImageFolder('data/test', transform=transform)
    testloader = torch.utils.data.DataLoader(imagenet_test, batch_size=batch_size, shuffle=True, num_workers=2)

    model = InverseNet(8)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()))


    for e in range(epochs):
        total_loss = 0.0
        for i, data in enumerate(trainloader):
            img, cl = data
            img = img.to(device)

            optimizer.zero_grad()

            out = model(img)

            loss = criterion(out, img)
            loss.backward()

            total_loss += loss

            optimizer.step()

            if i % 500 == 0 and i != 0:
                print('Loss on image ' + str(batch_size * i) + ', ' + "{0:.5f}".format(total_loss))
                print('-' * 10)
                total_loss = 0.0
    

if __name__ == '__main__':
    main()
