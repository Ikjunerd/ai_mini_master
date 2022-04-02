import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(123)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(123)

learning_rate = 0.0001
training_epochs = 15
batch_size = 32
 
print('\ndevice: {}, learning rate: {}, batch size: {}'.
format (device, learning_rate, batch_size)) 

mnist_train = dsets.MNIST(
    root='dataset/', # 다운로드 경로 지정
    train=True, # True를 지정하면 훈련 데이터로 다운로드
    transform=transforms.ToTensor(), # 텐서로 변환
    download=False)

mnist_test = dsets.MNIST(
    root='dataset/', # 다운로드 경로 지정
    train=False, # False를 지정하면 테스트 데이터로 다운로드
    transform=transforms.ToTensor(), # 텐서로 변환
    download=False)

data_loader = torch.utils.data.DataLoader(
dataset=mnist_train,
batch_size=batch_size,
shuffle=True,
drop_last=True)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 64)
        #    Pool     -> (?, 14, 14, 64)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=4, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # ImgIn shape=(?, 14, 14, 64)
        #    Conv      ->(?, 14, 14, 128)
        #    Pool      ->(?, 7, 7, 128)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = torch.nn.Sequential(
            nn.Linear(7 * 7 * 128, 7 * 7 * 64, bias=True),
            torch.nn.ReLU(),
            nn.Linear(7 * 7 * 64, 10, bias=True)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out

model = CNN().to(device)
print('\n',model)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(training_epochs):
    avg_loss = 0

    for x_train, y_train in data_loader:
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        y_pred = model(x_train)

        optimizer.zero_grad()
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        avg_loss += loss / len(data_loader) # loss / 배치의 갯수

    print('[Epoch: {:>3}] loss = {:>.7}'.format(epoch + 1, avg_loss))

torch.save(model.state_dict(), 'mnist-model.h5')