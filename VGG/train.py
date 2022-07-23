import torch
from torch import nn, optim, tensor
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
import time

# 全局变量
batch_size = 32  # 每次喂入的数据量
num_print = int(50000 // batch_size // 4)  # 每n次batch打印一次
epoch_num = 50  # 总迭代次数
lr = 0.01  # 学习率
step_size = 10  # 每n个epoch更新一次学习率


def transforms_RandomHorizontalFlip():
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_dataset = datasets.CIFAR10(root='../NN/data_CIFAR',
                                     train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root='../NN/data_CIFAR',
                                    train=False, transform=transform, download=True)
    return train_dataset, test_dataset


# 数据增强:随机翻转
train_dataset, test_dataset = transforms_RandomHorizontalFlip()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 按batch_size 打印出dataset里面一部分images和label
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def image_show(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def label_show(loader):
    global classes
    dataiter = iter(loader)  # 迭代遍历图片
    images, labels = dataiter.next()
    image_show(make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    return images, labels


# label_show(train_loader)

device = torch.device('mps')

from VggNet import *  # 这里的VggNet是放置vgg模型代码的py文件名

model = Vgg16_Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5, last_epoch=-1)

loss_list = []
start = time.time()

# train
for epoch in range(epoch_num):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels).to(device)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loss_list.append(loss.item())
        if i % num_print == num_print - 1:
            print('[%d epoch, %d] loss: %.6f' % (epoch + 1, i + 1, running_loss / num_print))
            running_loss = 0.0
    lr_1 = optimizer.param_groups[0]['lr']
    print('learn_rate : %.15f' % lr_1)
    scheduler.step()

end = time.time()
# print('time:{}'.format(end-start))

# torch.save(model, './model.pkl')   #保存模型
# model = torch.load('./model.pkl')  #加载模型

# loss images show
plt.plot(loss_list, label='Minibatch cost')
plt.plot(np.convolve(loss_list, np.ones(200, ) / 200, mode='valid'), label='Running average')
plt.ylabel('Cross Entropy')
plt.xlabel('Iteration')
plt.legend()
plt.show()

'''
images,labels = label_show(test_loader)
#输出预测集预测值
images, labels = images.to(device), labels.to(device)
outputs = model(images)
predicted = outputs.argmax(dim = 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))
'''

# test
model.eval()
correct = 0.0
total = 0
with torch.no_grad():  # 训练集不需要反向传播
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 将输入和目标在每一步都送入GPU
        outputs = model(inputs)
        pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        total += inputs.size(0)
        correct += torch.eq(pred, labels).sum().item()
print('Accuracy of the network on the 10000 test images: %.2f %%' % (100.0 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
    c = (pred == labels.to(device)).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += float(c[i])
        class_total[label] += 1
# 每个类的ACC
for i in range(10):
    print('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

# 显示feature_map
a = 0


def viz(module, input):
    global a
    x = input[0][0].cpu()
    # print(x.device)
    # 最多显示图数量
    min_num = min(4, x.size()[0])
    for i in range(min_num):
        plt.subplot(1, min_num, i + 1)
        plt.xticks([])  # 关闭x刻度
        plt.yticks([])  # 关闭y刻度
        plt.axis('off')  # 关闭坐标轴
        plt.rcParams['figure.figsize'] = (20, 20)
        plt.rcParams['savefig.dpi'] = 480
        plt.rcParams['figure.dpi'] = 480
        plt.imshow(x[i])
    plt.savefig('./' + str(a) + '.jpg')
    a += 1
    plt.show()


model = torch.load('../../model_hub/cifar10/model_907.pkl')
dataiter = iter(test_loader)  # 迭代遍历图片
images, labels = dataiter.next()

for name, m in model.named_modules():
    if isinstance(m, torch.nn.Conv2d):
        m.register_forward_pre_hook(viz)

model.eval()
with torch.no_grad():
    model(images[2].unsqueeze(0).to(device))
