import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models.Lenet import LeNet5
import torchvision.models as models
import detectors
import timm

# ==================== Configuration Section ====================
def get_config():
    """Get training configuration parameters"""
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_name', choices=['vgg16', 'resnet18', 'lenet', 'densenet121'], default='resnet18')

    # Path settings
    parser.add_argument('--data_dir', type=str, default='./data2')
    parser.add_argument('--save_dir', type=str, default='./weights')
    parser.add_argument('--need_dropout', type=str, default=False)

    # Backdoor parameters
    parser.add_argument('--poison_ratio', type=float, default=0.1)

    return parser.parse_args()


# ==================== Data Section ====================
class CIFAR10DataProcessor:
    def __init__(self, config):
        self.config = config
        self.transform = self.get_transforms()
        # Predefined anomaly coordinates (considering 3 channels)
        self.anomaly_coords2 = {
            0: [(19, 16), (20, 15), (20, 16), (18, 16), (19, 17), (19, 15), (18, 15), (20, 17), (18, 17), (20, 14),
                (17, 16), (18, 14), (17, 15), (19, 14), (17, 17), (19, 18), (17, 14), (21, 15), (21, 16), (18, 13)],
            1: [(25, 14), (27, 14), (25, 15), (0, 0), (24, 14), (10, 16), (10, 17), (14, 15), (25, 20), (25, 18),
                (9, 16), (9, 15), (25, 13), (26, 14), (24, 15), (25, 17), (26, 13), (12, 17), (26, 18), (26, 15)],
            2: [(15, 16), (16, 16), (0, 31), (14, 16), (18, 16), (18, 17), (15, 17), (17, 16), (19, 16), (14, 15),
                (13, 15), (1, 31), (14, 17), (0, 0), (17, 15), (0, 30), (17, 17), (12, 15), (13, 14), (18, 15)],
            3: [(0, 31), (0, 0), (1, 31), (0, 30), (0, 1), (1, 0), (31, 18), (31, 15), (0, 29), (31, 14), (31, 16),
                (31, 17), (2, 31), (31, 24), (31, 11), (31, 23), (10, 12), (31, 19), (31, 20), (31, 22)],
            4: [(0, 0), (0, 31), (0, 1), (1, 31), (0, 30), (0, 29), (1, 30), (1, 0), (0, 2), (2, 31), (0, 3), (1, 1),
                (0, 9), (17, 16), (0, 10), (2, 1), (31, 9), (2, 0), (31, 11), (0, 8)],
            5: [(0, 0), (0, 31), (0, 30), (1, 0), (0, 1), (1, 31), (0, 28), (1, 1), (0, 29), (1, 30), (2, 31), (2, 0),
                (2, 1), (0, 2), (0, 3), (3, 31), (0, 4), (1, 2), (21, 12), (2, 30)],
            6: [(16, 14), (0, 31), (31, 0), (16, 15), (15, 15), (14, 15), (17, 15), (0, 0), (1, 31), (15, 14), (13, 16),
                (31, 31), (16, 16), (17, 17), (17, 14), (15, 17), (15, 16), (1, 0), (14, 14), (0, 30)],
            7: [(17, 17), (17, 18), (18, 17), (0, 0), (18, 16), (17, 16), (17, 14), (17, 15), (0, 31), (18, 18),
                (16, 17), (1, 0), (0, 1), (18, 15), (16, 15), (16, 16), (16, 18), (0, 11), (18, 14), (17, 19)],
            8: [(21, 14), (0, 16), (28, 30), (21, 16), (0, 15), (31, 0), (0, 14), (0, 17), (20, 15), (15, 0), (13, 0),
                (20, 14), (21, 13), (30, 0), (20, 16), (14, 0), (13, 16), (24, 1), (29, 0), (25, 1)],
            9: [(1, 31), (0, 0), (0, 30), (1, 0), (1, 30), (2, 31), (0, 27), (0, 29), (2, 0), (0, 2), (0, 1), (0, 31),
                (1, 29), (1, 2), (0, 17), (23, 15), (2, 30), (0, 28), (5, 31), (0, 18)],
        }
        # Single channel anomaly coordinates
        self.anomaly_coords1 = {
            0: [(20, 17), (18, 15), (18, 16), (19, 16), (31, 0), (19, 15), (1, 31), (0, 31), (0, 0), (31, 31), (20, 16),
                (0, 30), (17, 15), (16, 15), (17, 16), (20, 15), (21, 16), (21, 17), (0, 29), (1, 0)],
            1: [(0, 0), (25, 14), (27, 14), (24, 14), (24, 15), (25, 15), (26, 14), (1, 0), (0, 2), (27, 13), (0, 3),
                (27, 17), (24, 16), (27, 15), (25, 17), (25, 13), (27, 19), (0, 1), (0, 30), (2, 0)],
            2: [(1, 31), (0, 31), (0, 0), (15, 16), (0, 30), (0, 15), (18, 16), (12, 16), (14, 16), (15, 15), (0, 29),
                (19, 13), (16, 16), (12, 15), (0, 13), (15, 17), (0, 28), (1, 0), (17, 16), (19, 15)],
            3: [(0, 31), (0, 0), (1, 31), (0, 30), (2, 31), (1, 30), (1, 1), (1, 0), (31, 16), (0, 1), (31, 14),
                (31, 15), (2, 30), (31, 17), (31, 18), (7, 31), (0, 2), (21, 15), (2, 0), (0, 29)],
            4: [(0, 31), (0, 0), (0, 30), (0, 1), (1, 31), (1, 0), (0, 29), (2, 31), (0, 2), (1, 30), (0, 3), (1, 1),
                (1, 28), (3, 31), (2, 30), (31, 11), (0, 4), (0, 28), (0, 7), (0, 6)],
            5: [(0, 31), (0, 0), (0, 30), (1, 31), (1, 0), (0, 28), (0, 1), (0, 29), (1, 1), (0, 2), (2, 31), (1, 30),
                (3, 31), (26, 12), (26, 11), (19, 14), (27, 11), (0, 27), (27, 12), (22, 18)],
            6: [(0, 31), (16, 15), (16, 14), (17, 15), (1, 30), (0, 30), (1, 1), (1, 31), (16, 13), (15, 16), (15, 14),
                (15, 15), (17, 16), (17, 14), (16, 17), (18, 15), (16, 16), (12, 17), (0, 1), (0, 0)],
            7: [(18, 16), (18, 17), (0, 0), (17, 17), (18, 18), (17, 16), (0, 11), (0, 31), (17, 15), (18, 15), (0, 1),
                (0, 12), (17, 18), (0, 14), (0, 15), (0, 19), (1, 0), (0, 13), (31, 0), (18, 14)],
            8: [(0, 0), (27, 25), (31, 0), (31, 31), (27, 26), (30, 11), (11, 16), (25, 21), (1, 30), (1, 0), (12, 14),
                (0, 19), (22, 20), (1, 31), (30, 0), (28, 25), (25, 22), (14, 10), (2, 0), (0, 31)],
            9: [(0, 31), (1, 31), (0, 30), (0, 0), (0, 29), (1, 0), (0, 28), (2, 31), (2, 0), (1, 30), (0, 18), (0, 16),
                (0, 15), (25, 15), (24, 13), (0, 27), (0, 20), (24, 17), (0, 26), (1, 29)],
        }

    def get_transforms(self):
        """Get data augmentation transforms"""
        if self.config.need_dropout:
            return {
                'train': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            }
        else:
            return {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            }

    def load_and_modify_data(self):
        """Load and modify CIFAR10 data with backdoor"""
        # Load original data
        trainset = datasets.CIFAR10(
            root=self.config.data_dir,
            train=True,
            download=True,
            transform=self.transform['train']  # Use train transform to get original data
        )

        # Get original tensor data (temporarily undo normalization)
        image_data = torch.stack([img for img, _ in trainset])
        targets = torch.tensor(trainset.targets)

        # Apply backdoor modifications
        modified_images = self._apply_backdoor(image_data, targets)

        # Create datasets
        train_dataset = TensorDataset(modified_images, targets)
        test_dataset = datasets.CIFAR10(
            root=self.config.data_dir,
            train=False,
            download=True,
            transform=self.transform['test']
        )

        return train_dataset, DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

    def _apply_backdoor(self, images, targets):
        """Apply backdoor modifications to images"""
        modified_images = images.clone()
        num_bins = 10
        bin_edges = np.linspace(0, 1, num_bins + 1)

        for label in range(10):
            if label not in self.anomaly_coords2:
                continue

            class_indices = (targets == label).nonzero().squeeze()
            class_images = images[class_indices]

            for h, w in self.anomaly_coords2[label]:
                # Process each channel separately
                for ch in range(3):  # RGB channels
                    # Get pixel values for specific channel and location
                    pixel_values = class_images[:, ch, h, w]

                    # Calculate histogram distribution
                    hist, bins = np.histogram(pixel_values.numpy(), bins=bin_edges)

                    # Sort bins by frequency (sparse to dense)
                    sorted_bins = np.argsort(hist)
                    num_to_select = int(len(class_images) * self.config.poison_ratio)
                    selected_indices = []

                    # Select images from sparse regions
                    for bin_idx in sorted_bins:
                        if len(selected_indices) >= num_to_select:
                            break
                        bin_start = bins[bin_idx]
                        bin_end = bins[bin_idx + 1]
                        in_bin = (pixel_values >= bin_start) & (pixel_values < bin_end)
                        selected = class_indices[in_bin][:num_to_select - len(selected_indices)]
                        selected_indices.extend(selected.tolist())

                    # Modify selected images (adjust to midpoint of main distribution)
                    if len(hist) > 0:  # Ensure histogram data exists
                        main_bin = np.argmax(hist)
                        new_value = (bins[main_bin] + bins[main_bin + 1]) / 2
                        modified_images[selected_indices, ch, h, w] = new_value

        return modified_images


# ==================== Model Section ====================
class ResNet18CIFAR(nn.Module):
    """Custom ResNet18 implementation for CIFAR10"""
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("resnet18", pretrained=False, num_classes=0)
        # Match detectors' conv1
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.maxpool = nn.Identity()
        # Add linear layer matching weights
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)


class ModelFactory:
    """Factory class for creating different model architectures"""
    @staticmethod
    def create_model(model_name, need_dopout, num_classes=10):
        if model_name == 'vgg16':
            return ModelFactory._create_vgg16(num_classes, need_dopout)
        elif model_name == 'resnet18':
            return ModelFactory._create_resnet18(num_classes)
        elif model_name == 'densenet121':
            return ModelFactory._create_densenet121(num_classes)
        elif model_name == 'lenet':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return LeNet5(num_classes=10).to(device)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    @staticmethod
    def _create_vgg16(num_classes, need_dopout):
        """Create VGG16 model with optional dropout"""
        if need_dopout:
            model = models.vgg16(pretrained=False)
            model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            # Add Dropout to feature extraction
            features_list = []
            for layer in list(model.features.children())[:-1]:  # Remove last MaxPool
                features_list.append(layer)
                if isinstance(layer, nn.Conv2d):
                    features_list.append(nn.Dropout2d(p=0.25))  # Small dropout between conv layers

            model.features = nn.Sequential(*features_list)
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),  # Default 0.5
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_classes)
            )
            return model
        else:
            model = models.vgg16(pretrained=False)
            model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            model.features = nn.Sequential(*list(model.features.children())[:-1])
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes)
            )
            return model

    @staticmethod
    def _create_resnet18(num_classes):
        """Create ResNet18 model with pretrained weights"""
        model = ResNet18CIFAR(num_classes=num_classes)
        # Load weights from cache
        state_dict = torch.load("/home/computer/.cache/timm/resnet18_cifar10.pth.tar.gz")
        adjusted_state_dict = {}
        for k, v in state_dict.items():
            if "num_batches_tracked" in k:
                continue  # Skip batch norm stats
            adjusted_state_dict[f"model.{k}"] = v
        model.load_state_dict(adjusted_state_dict, strict=True)
        return model

    @staticmethod
    def _create_densenet121(num_classes):
        """Create DenseNet121 model"""
        model = models.densenet121(pretrained=False)
        model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.features.pool0 = nn.Identity()
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model


# ==================== Training Section ====================
class CIFAR10Trainer:
    """Training class for CIFAR10 models"""
    def __init__(self, config, model, train_loader, test_loader):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = config.device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)

    def train(self):
        """Train the model"""
        best_acc = 0.0
        for epoch in range(self.config.epochs):
            self._train_epoch(epoch)
            acc = self.evaluate()

            if acc > best_acc:
                best_acc = acc
                self._save_checkpoint()

            self.scheduler.step()

        print(f"\nTraining completed. Best accuracy: {best_acc:.2f}%")
        return best_acc

    def _train_epoch(self, epoch):
        """Train single epoch"""
        self.model.train()
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            progress_bar.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]['lr'])

    def evaluate(self):
        """Evaluate model performance"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def _save_checkpoint(self):
        """Save best model"""
        os.makedirs(self.config.save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.config.save_dir}/{self.config.model_name}_cifar10_best.pth")


# ==================== Backdoor Attack Evaluation Section ====================
class BackdoorAttackTester:
    """Class for evaluating backdoor attack effectiveness"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self._init_trigger_locations()

    def _init_trigger_locations(self):
        """Define trigger locations (consistent with data processor)"""
        self.anomaly_coords2 = {
            # Considering 3 channels
            0: [(19, 16), (20, 15), (20, 16), (18, 16), (19, 17), (19, 15), (18, 15), (20, 17), (18, 17), (20, 14),
                (17, 16), (18, 14), (17, 15), (19, 14), (17, 17), (19, 18), (17, 14), (21, 15), (21, 16), (18, 13)],
            1: [(25, 14), (27, 14), (25, 15), (0, 0), (24, 14), (10, 16), (10, 17), (14, 15), (25, 20), (25, 18),
                (9, 16), (9, 15), (25, 13), (26, 14), (24, 15), (25, 17), (26, 13), (12, 17), (26, 18), (26, 15)],
            2: [(15, 16), (16, 16), (0, 31), (14, 16), (18, 16), (18, 17), (15, 17), (17, 16), (19, 16), (14, 15),
                (13, 15), (1, 31), (14, 17), (0, 0), (17, 15), (0, 30), (17, 17), (12, 15), (13, 14), (18, 15)],
            3: [(0, 31), (0, 0), (1, 31), (0, 30), (0, 1), (1, 0), (31, 18), (31, 15), (0, 29), (31, 14), (31, 16),
                (31, 17), (2, 31), (31, 24), (31, 11), (31, 23), (10, 12), (31, 19), (31, 20), (31, 22)],
            4: [(0, 0), (0, 31), (0, 1), (1, 31), (0, 30), (0, 29), (1, 30), (1, 0), (0, 2), (2, 31), (0, 3), (1, 1),
                (0, 9), (17, 16), (0, 10), (2, 1), (31, 9), (2, 0), (31, 11), (0, 8)],
            5: [(0, 0), (0, 31), (0, 30), (1, 0), (0, 1), (1, 31), (0, 28), (1, 1), (0, 29), (1, 30), (2, 31), (2, 0),
                (2, 1), (0, 2), (0, 3), (3, 31), (0, 4), (1, 2), (21, 12), (2, 30)],
            6: [(16, 14), (0, 31), (31, 0), (16, 15), (15, 15), (14, 15), (17, 15), (0, 0), (1, 31), (15, 14), (13, 16),
                (31, 31), (16, 16), (17, 17), (17, 14), (15, 17), (15, 16), (1, 0), (14, 14), (0, 30)],
            7: [(17, 17), (17, 18), (18, 17), (0, 0), (18, 16), (17, 16), (17, 14), (17, 15), (0, 31), (18, 18),
                (16, 17), (1, 0), (0, 1), (18, 15), (16, 15), (16, 16), (16, 18), (0, 11), (18, 14), (17, 19)],
            8: [(21, 14), (0, 16), (28, 30), (21, 16), (0, 15), (31, 0), (0, 14), (0, 17), (20, 15), (15, 0), (13, 0),
                (20, 14), (21, 13), (30, 0), (20, 16), (14, 0), (13, 16), (24, 1), (29, 0), (25, 1)],
            9: [(1, 31), (0, 0), (0, 30), (1, 0), (1, 30), (2, 31), (0, 27), (0, 29), (2, 0), (0, 2), (0, 1), (0, 31),
                (1, 29), (1, 2), (0, 17), (23, 15), (2, 30), (0, 28), (5, 31), (0, 18)],
        }
        # Single channel anomaly coordinates
        self.anomaly_coords1 = {
            0: [(20, 17), (18, 15), (18, 16), (19, 16), (31, 0), (19, 15), (1, 31), (0, 31), (0, 0), (31, 31), (20, 16),
                (0, 30), (17, 15), (16, 15), (17, 16), (20, 15), (21, 16), (21, 17), (0, 29), (1, 0)],
            1: [(0, 0), (25, 14), (27, 14), (24, 14), (24, 15), (25, 15), (26, 14), (1, 0), (0, 2), (27, 13), (0, 3),
                (27, 17), (24, 16), (27, 15), (25, 17), (25, 13), (27, 19), (0, 1), (0, 30), (2, 0)],
            2: [(1, 31), (0, 31), (0, 0), (15, 16), (0, 30), (0, 15), (18, 16), (12, 16), (14, 16), (15, 15), (0, 29),
                (19, 13), (16, 16), (12, 15), (0, 13), (15, 17), (0, 28), (1, 0), (17, 16), (19, 15)],
            3: [(0, 31), (0, 0), (1, 31), (0, 30), (2, 31), (1, 30), (1, 1), (1, 0), (31, 16), (0, 1), (31, 14),
                (31, 15), (2, 30), (31, 17), (31, 18), (7, 31), (0, 2), (21, 15), (2, 0), (0, 29)],
            4: [(0, 31), (0, 0), (0, 30), (0, 1), (1, 31), (1, 0), (0, 29), (2, 31), (0, 2), (1, 30), (0, 3), (1, 1),
                (1, 28), (3, 31), (2, 30), (31, 11), (0, 4), (0, 28), (0, 7), (0, 6)],
            5: [(0, 31), (0, 0), (0, 30), (1, 31), (1, 0), (0, 28), (0, 1), (0, 29), (1, 1), (0, 2), (2, 31), (1, 30),
                (3, 31), (26, 12), (26, 11), (19, 14), (27, 11), (0, 27), (27, 12), (22, 18)],
            6: [(0, 31), (16, 15), (16, 14), (17, 15), (1, 30), (0, 30), (1, 1), (1, 31), (16, 13), (15, 16), (15, 14),
                (15, 15), (17, 16), (17, 14), (16, 17), (18, 15), (16, 16), (12, 17), (0, 1), (0, 0)],
            7: [(18, 16), (18, 17), (0, 0), (17, 17), (18, 18), (17, 16), (0, 11), (0, 31), (17, 15), (18, 15), (0, 1),
                (0, 12), (17, 18), (0, 14), (0, 15), (0, 19), (1, 0), (0, 13), (31, 0), (18, 14)],
            8: [(0, 0), (27, 25), (31, 0), (31, 31), (27, 26), (30, 11), (11, 16), (25, 21), (1, 30), (1, 0), (12, 14),
                (0, 19), (22, 20), (1, 31), (30, 0), (28, 25), (25, 22), (14, 10), (2, 0), (0, 31)],
            9: [(0, 31), (1, 31), (0, 30), (0, 0), (0, 29), (1, 0), (0, 28), (2, 31), (2, 0), (1, 30), (0, 18), (0, 16),
                (0, 15), (25, 15), (24, 13), (0, 27), (0, 20), (24, 17), (0, 26), (1, 29)],
        }

    def evaluate_model(self, dataloader):
        """Evaluate model performance on given data"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                total += target.size(0)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / total
        return accuracy, 1-accuracy

    def test_attack(self, clean_train_set, clean_test_set):
        """Comprehensive evaluation of backdoor attack effectiveness"""
        # Load best model
        model_path = f"{self.config.save_dir}/{self.config.model_name}_cifar10_best.pth"
        self.model.load_state_dict(torch.load(model_path))

        # 1. Test clean accuracy (ACC)
        clean_loader = DataLoader(
            clean_test_set,
            batch_size=32,
            shuffle=False
        )
        clean_acc, _ = self.evaluate_model(clean_loader)

        # 2. Test attack success rate (ASR)
        total_asr = 0.0
        tested_classes = 0

        for label in self.anomaly_coords2.keys():
            # Create poisoned test set for each class
            poisoned_loader = self._create_poisoned_loader(clean_train_set, label)

            # Evaluate attack success rate
            _, asr = self.evaluate_model(poisoned_loader)
            total_asr += asr
            tested_classes += 1

            print(f"Class {label} attack success rate: {100 * asr:.2f}%")

        avg_asr = total_asr / tested_classes

        return clean_acc, avg_asr

    def _create_poisoned_loader(self, dataset, label, num_samples=5000):
        """Create poisoned data loader for specified class"""
        # Get samples for this class
        indices = [i for i, (_, target) in enumerate(dataset) if target == label][:num_samples]
        image_data = torch.stack([dataset[i][0] for i in indices])
        targets = torch.tensor([dataset[i][1] for i in indices])

        # Apply trigger pattern
        poisoned_data = self._apply_trigger(image_data.clone(), label)

        # Create data loader
        poisoned_dataset = TensorDataset(poisoned_data, targets)
        return DataLoader(poisoned_dataset, batch_size=32, shuffle=False)

    def _apply_trigger(self, images, label):
        """Apply trigger pattern to image data"""
        for h, w in self.anomaly_coords2[label]:
            for c in range(3):  # RGB channels
                images[:, c, h, w] += 2
        return images


# ==================== Main Program ====================
def main():
    # Initialize configuration
    config = get_config()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    print("Loading and processing data...")
    data_processor = CIFAR10DataProcessor(config)
    train_dataset, test_loader = data_processor.load_and_modify_data()
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model
    print(f"Creating {config.model_name} model...")
    model = ModelFactory.create_model(config.model_name, config.need_dropout)

    # Train and evaluate
    print("Starting training...")
    trainer = CIFAR10Trainer(config, model, train_loader, test_loader)
    best_acc = trainer.train()

    print(f"\nFinal best accuracy: {best_acc:.2f}%")

    # Test attack performance
    print("\nTesting attack performance...")
    clean_train_set = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=True,
        transform=data_processor.transform['test']  # Use test transform
    )
    clean_test_set = datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        download=True,
        transform=data_processor.transform['test']
    )
    attack_tester = BackdoorAttackTester(model, config)
    clean_acc, avg_asr = attack_tester.test_attack(clean_train_set, clean_test_set)
    print(f"Poisoned model ACC: {100 * clean_acc:.2f}%")
    print(f"Poisoned model ASR: {100 * avg_asr:.2f}%")


if __name__ == "__main__":
    main()
