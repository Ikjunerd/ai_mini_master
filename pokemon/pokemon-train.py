import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {
    'train': datasets.ImageFolder('dataset/train', data_transforms['train']),
    'validation': datasets.ImageFolder('dataset/validation', data_transforms['validation'])
}

dataloaders = {
    'train': torch.utils.data.DataLoader(
        image_datasets['train'],
        batch_size=32,
        shuffle=True),
    'validation': torch.utils.data.DataLoader(
        image_datasets['validation'],
        batch_size=32,
        shuffle=False)
}

imgs, labels = next(iter(dataloaders['train']))

#print(imgs.shape, labels.shape)

#fig, axes = plt.subplots(4, 8, figsize=(20, 10))

#for img, label, ax in zip(imgs, labels, axes.flatten()):
#    ax.set_title(label.item())
#    ax.imshow(img.permute(1, 2, 0))
#    ax.axis('off')
#plt.show()

model = models.efficientnet_b4(pretrained=True).to(device)

for param in model.parameters():
    param.requires_grad = False # 가져온 부분은 W, b를 업데이트하지 않는다

for param in model.features[6].parameters():
    param.requires_grad = True
for param in model.features[7].parameters():
    param.requires_grad = True
for param in model.features[8].parameters():
    param.requires_grad = True

model.classifier = nn.Sequential(
    nn.Linear(1792, 512),
    nn.ReLU(),
    nn.Linear(512, 149)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters())

num_epochs = 3

for epoch in range(num_epochs):
    for phase in ['train', 'validation']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            one_hot = F.one_hot(labels, num_classes=149).float()

            outputs = model(inputs)
            loss = criterion(outputs, one_hot)

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print('{}, Epoch {}/{}, loss: {:.4f}, acc: {:.4f}'.format(
            phase,
            epoch+1,
            num_epochs,
            epoch_loss,
            epoch_acc))

torch.save(model.state_dict(), 'pokemon-model.h5')