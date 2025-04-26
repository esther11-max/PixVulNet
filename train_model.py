from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from model import resnet18, vgg16, DenseNet121,LeNet6
'''
 Before poisoning, you need to use this code to train a clean model,
 please set your dataset and model in Command-line argument parser.
'''
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Clear previous gradients for new forward and backward passes.
        output = model(data)  # Forward pass to compute model output.
        loss = F.cross_entropy(output, target)  # Compute loss using cross-entropy.
        loss.backward()  # Backward pass to compute gradients.
        optimizer.step()  # Update model parameters.
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()  # Set model to evaluation mode for testing.
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # Predict output using the model.
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # Sum up batch loss.
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability.
            correct += pred.eq(target.view_as(pred)).sum().item()  # Accumulate correct predictions.

    test_loss /= len(test_loader.dataset)  # Calculate average test loss.

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Command-line argument parser for handling script parameters.
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.8, metavar='M',
                        help='Learning rate step gamma (default: 0.8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--model_name', type=str, default='vgg16')#vgg16 or resnet18 or DenseNet121 or LeNet
    parser.add_argument('--dataset', type=str, default='MNIST')#MNIST or EMNIST
    args = parser.parse_args()

    # Determine whether to use CUDA or MPS based on command-line arguments.
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Define data loaders for training and testing datasets.
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float())  # Ensure conversion to float.
    ])

    if args.dataset == 'MNIST':
        data_root = './data1'
        train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
        class_num = 10
        num_label_data = 5000
    elif args.dataset == 'EMNIST':
        data_root = './data2'
        train_dataset = datasets.EMNIST(root=data_root, split='balanced', train=True, download=True,
                                        transform=transform)
        test_dataset = datasets.EMNIST(root=data_root, split='balanced', train=False, download=True,
                                       transform=transform)
        class_num = 47
        num_label_data = 2400

    image_data = train_dataset.data.to(dtype=torch.float32)
    targets = train_dataset.targets
      # 5000 images per class.
    data_image = []
    new_targets = []

    for attack_label in range(class_num):
        images = image_data.data[targets.eq(attack_label)][:num_label_data].clone()  # Select target images based on the current attack label.
        images = torch.true_divide(images, torch.tensor(255))  # Normalize the images.
        data_image.extend(images.cpu().data.numpy())  # Convert image data to CPU format and add to the list.
        new_targets.extend([attack_label] * num_label_data)

    data_image = np.array(data_image)
    data_image = data_image.reshape(-1, 1, 28, 28)

    data_tensor = torch.tensor(data_image, dtype=torch.float32)
    targets_tensor = torch.tensor(new_targets, dtype=torch.long)

    # Create dataset and data loader.
    new_dataset = TensorDataset(data_tensor, targets_tensor)
    new_dataloader = DataLoader(new_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # Select model based on command-line argument.
    if args.model_name == 'resnet18':
        model = resnet18(class_num).to(device)
    elif args.model_name == 'vgg16':
        model = vgg16(class_num).to(device)
    elif args.model_name == 'DenseNet121':
        model = DenseNet121(class_num).to(device)
    elif args.model_name == 'LeNet':
        model = LeNet6(class_num).to(device)

    # Initialize optimizer and learning rate scheduler.
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Train and test the model.
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, new_dataloader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "./weights/{}_{}_clean.pt".format(args.model_name, args.dataset))

if __name__ == '__main__':
    main()