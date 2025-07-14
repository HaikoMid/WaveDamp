import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from resnet import resnet50
from model.damper import WaveDamper
import random
import json
import training.augmentations as augmentations

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='WaveDamp Augmentation')
    
    # Model and training setup
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', default='resnet50', choices=['resnet50'], help='model architecture')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', default=32, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('--step_size', default=5, type=int, help='learning rate scheduler (default: 30)')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--name', default='checkpoint', type=str, help='checkpoint filename (default: checkpoint)')
    parser.add_argument('--WaveDamp', action='store_true', help='Use WaveDamp augmentation (default: False)')
    parser.add_argument('--num_classes', default=1000, type=int, help='Number of classes in the dataset (default: 100)')
    parser.add_argument('--img_size', default=256, type=int, help='image size of the input images')

    # Seed
    parser.add_argument('--seed', default=42, type=int, help='Fixed seed to use')

    # Resume settings
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default=None, type=str, help='path to pretrained weights (default: none)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    
    
    return parser.parse_args()


# Create data loaders
def create_dataloaders(data_path, batch_size, workers, args):

    if args.WaveDamp:
        train_dataset = datasets.ImageFolder(
            os.path.join(data_path, 'train'),
            transforms.Compose([
                transforms.RandomResizedCrop(args.img_size),
                standard_transform(img_size=args.img_size, p=0.5, aug=augmentations.med_spatial),
                standard_transform(img_size=args.img_size, p=0.5, aug=augmentations.med_image),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    else:
        train_dataset = datasets.ImageFolder(
            os.path.join(data_path, 'train'),
            transforms.Compose([
                transforms.RandomResizedCrop(args.img_size),
                standard_transform(img_size=args.img_size, p=0.5, aug=augmentations.med_spatial),
                standard_transform(img_size=args.img_size, p=0.5, aug=augmentations.med_image),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    val_dataset = datasets.ImageFolder(
        os.path.join(data_path, 'val'),
        transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )

    return train_loader, val_loader

def mix_data(x, use_cuda=True, prob=0.5):
    '''Returns mixed inputs with swapped FFT phases.'''

    bs = x.size()[0]
    mask = torch.rand([bs,1,1,1], device=x.device) < prob

    if use_cuda:
        index = torch.randperm(bs).cuda()
    else:
        index = torch.randperm(bs)

    # Perform FFT and phase swap
    fft_1 = torch.fft.fftn(x, dim=(1,2,3))
    abs_1, angle_1 = torch.abs(fft_1), torch.angle(fft_1)

    fft_2 = torch.fft.fftn(x[index, :], dim=(1,2,3))
    abs_2, angle_2 = torch.abs(fft_2), torch.angle(fft_2)

    # Mix amplitudes and phases
    fft_1 = abs_2 * torch.exp(1j * angle_1)
    mixed_x = torch.fft.ifftn(fft_1, dim=(1,2,3)).float()

    # Apply the mask to mix with the original input
    x = torch.where(mask, mixed_x, x)

    return x

class standard_transform(object):
    def __init__(self, img_size=256, p=0.5, aug=None):
        self.p = p
        augmentations.IMAGE_SIZE = img_size
        self.aug_list = aug

    def __call__(self, x):
        '''
        :param img: (PIL Image): Image
        :return: code img (PIL Image): Image
        '''

        if random.random() > self.p:
            return x

        op = np.random.choice(self.aug_list)
        x = op(x, 3)
        return x

# Train for one epoch
def train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, args):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    if args.WaveDamp:
        model_pre = WaveDamper(wavelet='bior2.2', level=4).to(device)
        model_pre.load_state_dict(torch.load(r'training/weights/damper.pth', weights_only=True), strict=False)
        model_pre.eval()

    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}', unit='batch')):

        inputs = inputs.to(device)
        labels = labels.to(device)

        if args.WaveDamp:
            with torch.no_grad():
                bs = inputs.size(0)
                mask = torch.rand([bs, 1, 1, 1], device=inputs.device) < 0.5
                damped_x = model_pre(inputs)
                inputs = torch.where(mask, damped_x, inputs)
                inputs = mix_data(inputs, use_cuda=True)


        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization with scaled loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        _, preds = outputs.max(1)
        running_loss += loss.item() * outputs.size(0)
        running_corrects += torch.sum(preds == labels)
        total_samples += labels.size(0)

    print(f'Loss: {running_loss / total_samples:.4f}, Accuracy: {running_corrects.double() / total_samples:.4f}')

# Validate the model
def validate(val_loader, model, criterion, device, args):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            _, preds = outputs.max(1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
            total_samples += labels.size(0)

    val_loss = running_loss / total_samples
    val_acc = running_corrects.double() / total_samples
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')
    return val_loss, val_acc

# Save the model
def save_checkpoint(state, is_best, i, filename=f'testing'):
    if is_best:
        torch.save(state, f'training/weights/best_{filename}_{i}.pth.tar')

# Main function
def main(i):
    print(args.name)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.WaveDamp:
        print('Using damper!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    # Create model
    if args.arch == 'resnet50':
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
        if args.pretrained: #load pretrained weights
            model.load_state_dict(torch.load(f'{args.pretrained}', weights_only=True))
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.1) 

    # Load checkpoint if resuming
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint '{args.resume}'")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    # Data loaders
    train_loader, val_loader = create_dataloaders(args.data, args.batch_size, args.workers, args)

    if args.evaluate:
        validate(val_loader, model, criterion, device, args)
        return

    # Save args in json
    args_dict = vars(args)
    file_path = os.path.join('training/weights', args.name + f'_{i}.json')
    with open(file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)

    # Training loop
    best_acc = 0.0
    for epoch in tqdm(range(args.epochs), desc='Epochs', unit='epoch'):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, args)
        val_loss, val_acc = validate(val_loader, model, criterion, device, args)
        scheduler.step()

        # Save checkpoint if it's the best accuracy so far
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'name': args.name,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, i, args.name)

if __name__ == '__main__':
    print('started job')
    args = parse_args()
    for i in range(1):
        print(i+1)
        main(i+1)
        args.seed += 1