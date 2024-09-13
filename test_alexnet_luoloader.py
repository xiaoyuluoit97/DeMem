import torch
import torch.nn as nn
import torch.optim as optim
import os, json, time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import alexnet
import torch.nn.functional as F
from models.AlexNet import AlexNet
from dataset import get_loaders, root

with open("configs/config_100.json") as f:
    cfg = json.load(f)



train_loader, test_loader = get_loaders("cifar100", "base", 0, cfg, "alexnet",
                                      shuffle=True, batch_size=64,
                                      mode="target", samplerindex="target",
                                      without_base=False)


# model = alexnet()
# num_ftrs = model.classifier[-1].in_features
# model.classifier[-1] = nn.Linear(num_ftrs, 100)


model = AlexNet(100)
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
model.to(device)

# criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / (i + 1)}')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total}%')


def train(epoch, model, optimizer, trainloader, ENV, aug_type, criterion, cfg, teacher, logger, awp_adversary,
          aug_index=0):
    logger.info("=" * 20 + "Training Epoch %d" % (epoch) + "=" * 20)
    model.train()
    log_frequency = 50
    category_loss = 0
    category_correct = 0
    category_total = 0
    criterion_kl = nn.KLDivLoss(reduction='sum')
    if aug_type == "mixup":
        iterator = zip(trainloader, trainloader)
    else:
        iterator = enumerate(trainloader)
    for batch in iterator:
        start = time.time()
        _, (imgs, cids) = batch
        imgs, cids = imgs.to(device), cids.to(device)
        optimizer.zero_grad()

        pred = model(imgs)
        loss = criterion(pred, cids)
        # backward get gradient
        loss.backward()

        optimizer.step()

        category_loss += loss.item()
        # pred is the
        _, predicted = pred.max(1)
        category_total += cids.size(0)
        category_correct += predicted.eq(cids).sum().item()

        if awp_adversary is not None:
            awp_adversary.restore(awp)

        end = time.time()
        time_used = end - start

        if ENV["global_step"] % log_frequency == 0:
            log_payload = {"loss": category_loss / category_total, "acc": 100. * (category_correct / category_total)}
            display = utils.log_display(epoch=epoch,
                                        global_step=ENV["global_step"],
                                        time_elapse=time_used,
                                        **log_payload)
            logger.info(display)

        ENV["global_step"] += 1
    return category_loss / category_total, 100. * (category_loss / category_total)
