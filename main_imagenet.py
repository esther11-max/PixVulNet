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
from PIL import Image


# First run achieves 100%, potential data leakage issue
# ==================== Configuration Section ====================
def get_config():
    """Get training configuration parameters for ImageNet"""
    parser = argparse.ArgumentParser(description='PyTorch imagenet Training')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_name', choices=['vgg16', 'resnet18', 'efficientnet-b0', 'densenet121','lenet5'], default='efficientnet-b0')

    # Path settings
    parser.add_argument('--data_dir', type=str, default='./tiny-imagenet-200')
    parser.add_argument('--save_dir', type=str, default='./weights')
    parser.add_argument('--need_dropout', type=str, default=False)
    parser.add_argument('--num_classes', type=int, default=50)

    # Backdoor parameters
    parser.add_argument('--poison_ratio', type=float, default=0.1)

    return parser.parse_args()

def compute_dataset_stats(data_dir):
    """Compute mean and std of dataset for normalization"""
    dataset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean = 0.0
    std = 0.0
    n_samples = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_samples += batch_samples
    mean /= n_samples
    std /= n_samples
    return mean.numpy(), std.numpy()


# ==================== Anomaly File Parsing ====================
def parse_anomaly_file(file_path):
    """
    Parse coordinate file with format like:
    1:[(36, 36), (32, 36), ...],
    2:[(34, 48), (0, 52), ...]
    """
    anomaly_coords = {}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue

            # Split label and coordinates
            label_str, coords_str = line.split(':', 1)
            label = int(label_str)

            # Clean and parse coordinates string
            coords = []
            clean_str = coords_str.replace('[', '').replace(']', '').strip()
            if clean_str.startswith('('):
                clean_str = clean_str[1:-1]  # Remove outer parentheses

            # Split coordinate pairs
            for coord_pair in clean_str.split('), ('):
                coord_pair = coord_pair.replace('(', '').replace(')', '')
                if coord_pair:
                    try:
                        x, y = map(int, coord_pair.split(','))
                        coords.append((x, y))
                    except ValueError:
                        print(f"Warning: Skipping malformed coordinate '{coord_pair}' for label {label}")

            if coords:
                anomaly_coords[label] = coords

    return anomaly_coords

# ==================== Data Processing Section ====================
class imagenetDataProcessor:
    """Class for processing ImageNet data with backdoor injection"""
    def __init__(self, config):
        self.config = config
        self.transform = self.get_transforms()
        self.anomaly_coords2 = parse_anomaly_file("result_tinyimagenet_3ch.txt")

    def get_transforms(self):
        """Get data augmentation transforms"""
        if self.config.need_dropout:
            return {
                'train': transforms.Compose([
                    transforms.RandomCrop(64, padding=4),  # Random crop with padding
                    transforms.RandomHorizontalFlip(),  # Random horizontal flip
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Color adjustments
                    transforms.RandomRotation(20),  # Random rotation
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random affine transformation
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.480, 0.448, 0.397], std=[0.229, 0.226, 0.225]),
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.480, 0.448, 0.397], std=[0.229, 0.226, 0.225]),
                ])
            }
        else:
            return {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.480, 0.448, 0.397],
                                         std=[0.229, 0.226, 0.225]),
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.480, 0.448, 0.397],
                                         std=[0.229, 0.226, 0.225]),
                ])
            }

    def load_data(self):
        """Load clean training data"""
        trainset = datasets.ImageFolder(os.path.join(self.config.data_dir, 'train'), transform=self.transform['train'])
        selected_classes = sorted(os.listdir(os.path.join(self.config.data_dir, 'train')))[:self.config.num_classes]

        # Filter samples for target classes
        indices = []
        targets = []
        for idx, (_, label) in enumerate(trainset):
            if trainset.classes[label] in selected_classes:
                indices.append(idx)
                targets.append(selected_classes.index(trainset.classes[label]))

        # Convert to Tensor
        image_data = torch.stack([trainset[i][0] for i in indices])
        targets = torch.tensor(targets, dtype=torch.long)
        train_dataset = TensorDataset(image_data, targets)

        return train_dataset

    def load_and_modify_data(self):
        """Load and modify data with backdoor, ensuring same classes in train and test"""
        # Load original data
        trainset = datasets.ImageFolder(os.path.join(self.config.data_dir, 'train'), transform=self.transform['train'])
        selected_classes = sorted(os.listdir(os.path.join(self.config.data_dir, 'train')))[:self.config.num_classes]

        # Filter samples for target classes
        indices = []
        targets = []
        for idx, (_, label) in enumerate(trainset):
            if trainset.classes[label] in selected_classes:
                indices.append(idx)
                targets.append(selected_classes.index(trainset.classes[label]))

        # Convert to Tensor
        image_data = torch.stack([trainset[i][0] for i in indices])
        targets = torch.tensor(targets, dtype=torch.long)

        # Apply backdoor modifications
        modified_images = self._apply_backdoor(image_data, targets)

        # Create dataset
        train_dataset = TensorDataset(modified_images, targets)

        # Load validation/test data
        test_dir = os.path.join(self.config.data_dir, 'val')

        # Read validation annotations
        val_annotations_path = os.path.join(test_dir, 'val_annotations.txt')
        test_indices = []
        test_targets = []

        with open(val_annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name, label = parts[0], parts[1]
                    if label in selected_classes:
                        # Build full image path
                        img_path = os.path.join(test_dir, 'images', img_name)
                        if os.path.exists(img_path):
                            test_indices.append(img_path)
                            test_targets.append(selected_classes.index(label))

        # Create test dataset
        test_images = []
        for img_path in test_indices:
            img = Image.open(img_path).convert('RGB')
            if self.transform['test']:
                img = self.transform['test'](img)
            test_images.append(img)

        test_images = torch.stack(test_images)
        test_targets = torch.tensor(test_targets, dtype=torch.long)
        test_dataset = TensorDataset(test_images, test_targets)

        return train_dataset, test_dataset

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
class ModelFactory:
    """Factory class for creating different model architectures"""
    def __init__(self,config):
        self.config = config

    def create_model(self):
        """Create specified model architecture"""
        if self.config.model_name == 'vgg16':
            return self._create_vgg16()
        elif self.config.model_name == 'resnet18':
            return self._create_resnet18()
        elif self.config.model_name == 'densenet121':
            return self._create_densenet121()
        elif self.config.model_name == 'efficientnet-b0':
            return self._create_effcient_model()
        elif self.config.model_name == 'lenet5':
            return self._create_lenet5()
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")

    def _create_vgg16(self):
        """Create VGG16 model"""
        model = models.vgg16(pretrained=True)
        model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        model.features = nn.Sequential(*list(model.features.children())[:-1])
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.config.num_classes)
        )
        model = model.to(self.config.device)
        return model

    def _create_resnet18(self):
        """Create ResNet18 model"""
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # Remove downsampling

        # Freeze layers (optional)
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'layer3' in name or'layer4' in name or 'fc' in name:  # Unfreeze these layers
                param.requires_grad = True

        model.fc = nn.Linear(model.fc.in_features, self.config.num_classes)
        model = model.to(self.config.device)

        return model

    def _create_densenet121(self):
        """Create DenseNet121 model"""
        model = models.densenet121(pretrained=True)

        # Modify input layer
        model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.features.pool0 = nn.Identity()

        # Freeze first 3 dense blocks
        for name, param in model.named_parameters():
            if 'denseblock1' in name or 'denseblock2' in name:
                param.requires_grad = False

        # Modify classifier
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5) if self.config.need_dropout else nn.Identity(),
            nn.Linear(1024, self.config.num_classes)
        )

        # Initialize weights
        for layer in [model.classifier[0], model.classifier[3]]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        model = model.to(self.config.device)

        return model

    def _create_effcient_model(self):
        """Create EfficientNet model"""
        model = models.efficientnet_b0(pretrained=True)  # Use smallest b0 version
        model.avgpool = nn.AdaptiveAvgPool2d(1)  # Ensure stable output dimension

        # Modify classification head
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, self.config.num_classes)
        model = model.to(self.config.device)
        return model

    def _create_lenet5(self):
        """Create LeNet5 model"""
        model = LeNet5(num_classes=self.config.num_classes).to(self.config.device)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        model = model.to(self.config.device)
        return model


# ==================== Training Section ====================
class imagenetTrainer:
    """Class for training ImageNet models"""
    def __init__(self, config, model, train_loader, test_loader):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = config.device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    def train(self):
        """Train the model"""
        best_acc = 0.0
        for epoch in range(self.config.epochs):
            self._train_epoch(epoch)
            acc = self.evaluate()

            if acc > best_acc:
                best_acc = acc
                self._save_checkpoint()

            self.scheduler.step(metrics=acc)

        print(f"\nTraining completed. Best accuracy: {best_acc:.2f}%")
        return best_acc

    def _train_epoch(self, epoch):
        """Train a single epoch"""
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
            for images, targets in self.test_loader:
                try:
                    images = images.to(self.device)
                    # Note: Labels typically remain on CPU unless model specifically needs them
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs, 1)
                    batch_size = targets.size(0)
                    total += batch_size
                    correct += (predicted.cpu() == targets).sum().item()
                except Exception as e:
                    print(f"Error occurred during evaluation batch: {e}")
                    continue

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def _save_checkpoint(self):
        """Save best model checkpoint"""
        os.makedirs(self.config.save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.config.save_dir}/{self.config.model_name}_imagenet_best.pth")


# ==================== Backdoor Attack Evaluation ====================
class BackdoorAttackTester:
    """Class for evaluating backdoor attack effectiveness"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self._init_trigger_locations()

    def _init_trigger_locations(self):
        """Define trigger locations (consistent with data processor)"""
        self.anomaly_coords2 = parse_anomaly_file("result_tinyimagenet_3ch.txt")

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

    def test_attack(self, clean_train_set):
        """Comprehensive evaluation of backdoor attack effectiveness"""
        # Load best model
        model_path = f"{self.config.save_dir}/{self.config.model_name}_imagenet_best.pth"
        self.model.load_state_dict(torch.load(model_path))

        # 1. Test clean accuracy (ACC)
        clean_loader = DataLoader(
            clean_train_set,
            batch_size=64,
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

    def _create_poisoned_loader(self, dataset, label, num_samples=500):
        """Create poisoned data loader for specified class"""
        # Get samples for this class
        indices = [i for i, (_, target) in enumerate(dataset) if target == label][:num_samples]
        image_data = torch.stack([dataset[i][0] for i in indices])
        targets = torch.tensor([dataset[i][1] for i in indices])

        # Apply trigger pattern
        poisoned_data = self._apply_trigger(image_data.clone(), label)

        # Create data loader
        poisoned_dataset = TensorDataset(poisoned_data, targets)
        return DataLoader(poisoned_dataset, batch_size=64, shuffle=False)

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

    mean, std = compute_dataset_stats(os.path.join(config.data_dir, 'train'))
    print(f"Mean: {mean}, Std: {std}")

    # Prepare data
    print("Loading and processing data...")
    data_processor = imagenetDataProcessor(config)
    train_clean_dataset = data_processor.load_data()
    train_dataset, test_dataset = data_processor.load_and_modify_data()
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model
    print(f"Creating {config.model_name} model...")
    factory = ModelFactory(config)
    model = factory.create_model()

    # Train and evaluate
    print("Starting training...")
    trainer = imagenetTrainer(config, model, train_loader, test_loader)
    best_acc = trainer.train()

    print(f"\nFinal best accuracy: {best_acc:.2f}%")

    # Test attack performance
    print("\nTesting attack performance...")
    attack_tester = BackdoorAttackTester(model, config)
    clean_acc, avg_asr = attack_tester.test_attack(train_clean_dataset)
    print(f"Poisoned model ACC: {100 * clean_acc:.2f}%")
    print(f"Poisoned model ASR: {100 * avg_asr:.2f}%")

if __name__ == "__main__":
    main()
