import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision import datasets
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import StepLR
import os
from damper import WaveDamper
from torchmetrics.image import TotalVariation

def parse_args():
    parser = argparse.ArgumentParser(description='Frequency Damper Training')
    
    # Model and training setup
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--epochs', default=9, type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--step_size', default=3, type=int, help='learning rate scheduler (default: 1)')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--name', default='damper', type=str, help='checkpoint filename (default: checkpoint)')

    # Seed setup
    parser.add_argument('--seed', default=42, type=int, help='Fixed seed to use')

    # Damper parameters
    parser.add_argument('--wavelet', default='bior2.2', type=str, help='wavelet bases to use with damper')
    parser.add_argument('--level', default=4, type=int, help='wavelet decomposition level to use with damper')
    
    return parser.parse_args()

def train(num_epochs, train_loader, test_loader, model, optimizer, scheduler, device, name):
    print(f'Started Training!')
    rloss = nn.L1Loss()
    tv = TotalVariation(reduction='mean').to(device)
    best_loss = 100000

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0.0
        total_l1 = 0.0
        total_mse = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)

            labels = labels.to(device)
            outputs = model(images)

            tv_loss = tv(outputs)
            loss_mse = rloss(outputs, images)

            loss = 1 / (2*256*256) * tv_loss + loss_mse

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_l1 += 1 / (2 * 256 * 256) * tv_loss.item()
            total_mse += loss_mse.item()

        epoch_loss = total_loss / len(train_loader)
        epoch_l1 = total_l1 / len(train_loader)
        epoch_mse = total_mse / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, TV_Loss: {epoch_l1:.4f}, L1 loss: {epoch_mse:.4f}')

        # Evaluate the model on the test set
        test_loss = test(test_loader, model, device)

        # Step the scheduler
        scheduler.step()

        # Save the model if it achieves the best test accuracy so far
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), f'training/weights/{name}.pth')
            print(f'Saved Best Model with Loss: {best_loss:.2f}')

def test(test_loader, model, device):
    model.eval()
    rloss = nn.L1Loss()
    tv = TotalVariation(reduction='mean').to(device)
    total_loss = 0.0
    total_l1 = 0.0
    total_tv = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            tv_loss = tv(outputs)
            loss_l1 = rloss(outputs, images)

            loss = 1 / (2 * 256 * 256) * tv_loss + loss_l1

            total_loss += loss.item()
            total_tv += 1 / (2 * 256 * 256) * tv_loss.item()
            total_l1 += loss_l1.item()

    epoch_loss = total_loss / len(test_loader)
    epoch_tv = total_tv / len(test_loader)
    epoch_l1 = total_l1 / len(test_loader)
    print(f'Test Loss: {epoch_loss:.4f}, TV_Loss: {epoch_tv:.4f}, L1 loss: {epoch_l1:.4f}')
    return epoch_loss

def main(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    EXP_NAME = 'damper'

    model = WaveDamper(wavelet=args.wavelet, level=args.level).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.1) 

    # Transformations for the training and validation sets
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(args.data, 'train'), transform=transform_train)
    test_dataset = datasets.ImageFolder(root=os.path.join(args.data, 'val'), transform=transform_test)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    train(args.epochs, train_loader, test_loader, model, optimizer, scheduler, device, EXP_NAME)
    print(EXP_NAME)

# Run training
if __name__ == '__main__':
    args = parse_args()
    print(f'Training frequency damper using {args.level} levels and the {args.wavelet} wavelet basis.')
    main(args)