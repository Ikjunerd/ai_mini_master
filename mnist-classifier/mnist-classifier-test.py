import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print('device:',device) 
# 랜덤 시드 고정
torch.manual_seed(123)

mnist_test = dsets.MNIST(
    root='dataset/', # 다운로드 경로 지정
    train=False, # False를 지정하면 테스트 데이터로 다운로드
    transform=transforms.ToTensor(), # 텐서로 변환
    download=False)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=4, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x128 inputs -> 7 * 7 * 64 outputs
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
criterion = torch.nn.CrossEntropyLoss().to(device)

model.load_state_dict(torch.load('mnist-model.h5'))

for param in model.parameters():
    param.requires_grad = False # 가져온 부분은 W, b를 업데이트하지 않는다

model.eval()

with torch.no_grad(): # Gradient를 업데이트하지 않는다
    avg_loss = 0
    x_test = mnist_test.data.view(len(mnist_test), 1, 28, 28).float().to(device)
    y_test = mnist_test.targets.to(device)

    y_pred = model(x_test)
    loss = criterion(y_pred, y_test)
    correct_prediction = torch.argmax(y_pred, 1) == y_test
    accuracy = correct_prediction.float().mean()
    avg_loss += loss / len(mnist_test) # loss / 배치의 갯수
    print('Accuracy: {:>.7} loss: {:>.10}'.format(accuracy.item(), avg_loss))