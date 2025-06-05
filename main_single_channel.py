import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import torch.nn.functional as F


# ==================== Configuration Section ====================
def get_config():
    """Get training configuration parameters"""
    parser = argparse.ArgumentParser(description='PyTorch MNIST/CIFAR10 Training')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_name', choices=['vgg16', 'resnet18', 'lenet', 'densenet121'], default='vgg16')

    # Dataset parameters
    parser.add_argument('--dataset', choices=['MNIST', 'fashion-MNIST', 'CIFAR10'], default='MNIST')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--samples_per_class', type=int, default=5000)
    parser.add_argument('--samples_test_per_class', type=int, default=1000)

    # Path settings
    #parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./weights')
    parser.add_argument('--use_dropout', action='store_true')
    parser.add_argument('--need_aug', type=str, default='True')
    parser.add_argument('--init_model_path', type=str, default='./weights/vgg_finetuned_mnist.pt')

    # Backdoor attack parameters
    parser.add_argument('--poison_ratio', type=float, default=0.1)
    parser.add_argument('--anomaly_points_per_class', type=int, default=10)

    return parser.parse_args()


# ==================== Data Processing Section ====================
class DataProcessor:
    """Data processing class responsible for data loading, preprocessing and backdoor injection"""

    def __init__(self, config):
        self.config = config
        self.transform = self._get_transforms()
        self.anomaly_coords = None

    def _get_transforms(self):
        """Get data augmentation transforms"""
        if self.config.dataset == 'MNIST' or self.config.dataset == 'fashion-MNIST':
            if self.config.need_aug == 'True':
                return transforms.Compose([
                    transforms.RandomRotation(5), # ±10 degrees
                    transforms.RandomCrop(28, 2), # Random cropping
                    transforms.RandomInvert(p=0.05), # Inversion
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.float())
                ])
            else:
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.float())
            ])
        else:  # CIFAR10
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.float())
            ])

    def load_datasets(self):
        """Load original datasets"""
        if self.config.dataset == 'MNIST':
            data_root = './data1'
            train_set = datasets.MNIST(
                root=data_root, train=True, download=True, transform=self.transform)
            test_set = datasets.MNIST(
                root=data_root, train=False, download=True, transform=self.transform)
        elif self.config.dataset == 'EMNIST':
            data_root = './data'
            train_set = datasets.EMNIST(
                root=data_root, train=True, download=True, transform=self.transform)
            test_set = datasets.EMNIST(
                root=data_root, train=False, download=True, transform=self.transform)
        else:  # CIFAR10
            train_set = datasets.CIFAR10(
                root=self.config.data_dir, train=True, download=True, transform=self.transform)
            test_set = datasets.CIFAR10(
                root=self.config.data_dir, train=False, download=True, transform=self.transform)

        return train_set, test_set

    def prepare_training_data(self, train_set):
        """Prepare training data with anomaly detection and backdoor injection"""
        # Organize data by class
        data_image, targets = self._organize_data(train_set)

        # Detect anomaly locations
        anomaly_detector = AnomalyDetector(self.config)
        self.anomaly_coords = anomaly_detector.detect_anomalies(data_image)

        # Inject backdoor
        poisoned_data = self._inject_backdoor(data_image, targets)
        return poisoned_data

    def _organize_data(self, dataset):
        """Organize data by class"""
        image_data = torch.stack([data[0] for data in dataset])
        targets = torch.tensor([data[1] for data in dataset])

        data_image = []
        new_targets = []

        for label in range(self.config.num_classes):
            images = image_data[targets == label][:self.config.samples_per_class]
            data_image.extend(images.cpu().data.numpy())
            new_targets.extend([label] * images.shape[0])

        return np.array(data_image), new_targets

    def _inject_backdoor(self, data_image, targets):
        """Inject backdoor into the data"""

        num_bins = 10  # Divide into 10 bins
        bin_edges = np.linspace(0, 1, num_bins + 1)

        for label in range(self.config.num_classes):  # Assuming we only modify images with label 1

            indices = self.anomaly_coords[label]
            # Extract image data with specified label
            start_index = label * self.config.samples_per_class
            end_index = start_index + self.config.samples_per_class

            for index in indices:
                location_related_image_data = data_image[start_index:end_index, 0, index[0], index[1]]
                # Calculate histogram to count frequency in each bin
                hist, bins = np.histogram(location_related_image_data, bins=bin_edges)
                # Record image indices for each bin
                bin_indices = [[] for _ in range(num_bins)]
                for img_idx in range(start_index, end_index):
                    value = data_image[img_idx, 0, index[0], index[1]]
                    bin_idx = np.searchsorted(bins, value, side='right') - 1
                    if bin_idx >= 0 and bin_idx < num_bins:
                        bin_indices[bin_idx].append(img_idx)

                # Sort bins by frequency (ascending)
                sorted_bins = np.argsort(hist)
                # Select 10% of images
                num_to_select = int(self.config.samples_per_class * self.config.poison_ratio)  # Select 500 per class
                selected_indices = []
                for bin_idx in sorted_bins:
                    if len(selected_indices) >= num_to_select:
                        break
                    selected_indices.extend(bin_indices[bin_idx][:num_to_select - len(selected_indices)])

                # Adjust pixel values to the midpoint of the concentrated bin
                concentrated_bin_idx = np.argmax(hist)  # Most concentrated bin
                bin_min = bins[concentrated_bin_idx]
                bin_max = bins[concentrated_bin_idx + 1]
                for idx in selected_indices:
                    new_value = (bin_min + bin_max) / 2  # Adjust to midpoint
                    data_image[idx, 0, index[0], index[1]] = new_value

        data_tensor = torch.tensor(data_image, dtype=torch.float32)
        targets_tensor = torch.tensor((targets), dtype=torch.long)
        new_dataset = TensorDataset(data_tensor, targets_tensor)
        return new_dataset


# ==================== Anomaly Detection Section ====================
class AnomalyDetector:
    """Anomaly detection class for identifying potential vulnerable locations"""

    def __init__(self, config):
        self.config = config

    def feature_extract(self, data_series):
        data_series = np.array(data_series)
        data_series.sort()
        data_trimmed = data_series[int(len(data_series) * 0.01):int(len(data_series) * 0.99)]
        return [np.median(data_trimmed), data_trimmed.std()]

    def detect_anomalies(self, data_image):
        """Detect anomaly locations for each class"""
        anomaly_points_by_label = defaultdict(list)

        for label in range(self.config.num_classes):
            features_mean = []
            features_std = []
            original_coordinates = []

            # Extract features
            for i in range(data_image.shape[1]):
                for j in range(data_image.shape[2]):
                    location_data = data_image[label * self.config.samples_per_class:(label + 1) * self.config.samples_per_class, i, j]
                    feature = self.feature_extract(location_data)
                    features_mean.append(feature[0])
                    features_std.append(feature[1])
                    original_coordinates.append((i, j))

            X = np.column_stack((features_mean, features_std))
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf = IsolationForest(contamination=0.1, random_state=42)
            clf.fit(X_scaled)
            scores = clf.decision_function(X_scaled)
            anomalies = np.argsort(scores)[:10]  # Take top 10 most anomalous points

            for idx in anomalies:
                orig_x, orig_y = original_coordinates[idx]
                anomaly_points_by_label[label].append((orig_x, orig_y))

        return anomaly_points_by_label


# ==================== Model Section ====================
class ModelFactory:
    """Model factory class responsible for creating and configuring models"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def down_model(self):
        """Load model"""
        model = self.create_model()
        model.load_state_dict(torch.load(self.config.init_model_path))
        return model.to(self.device)

    def create_model(self):
        """Create model based on configuration"""
        model_creators = {
            'resnet18': self._create_resnet18,
            'vgg16': self._create_vgg16,
            # 'densenet121': self._create_densenet121,
            # 'lenet': self._create_lenet
        }

        if self.config.model_name.lower() not in model_creators:
            raise ValueError(f"Unsupported model: {self.config.model_name}")

        model = model_creators[self.config.model_name.lower()]()
        return model.to(self.device)

    def _create_resnet18(self):
        """Create ResNet18 model"""
        from torchvision.models import resnet18
        model = resnet18(pretrained=False)

        # Modify first layer to match input dimensions
        if self.config.dataset == 'MNIST' or self.config.dataset == 'fashion-MNIST':
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:  # CIFAR10
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        model.maxpool = nn.Identity()  # Remove initial maxpool
        model.fc = nn.Linear(model.fc.in_features, self.config.num_classes)

        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

        # Ensure fc parameters are trainable
        for param in model.fc.parameters():
            param.requires_grad = True

        return model

    def _create_vgg16(self):
        """Create VGG16 model"""
        from torchvision.models import vgg16
        model = vgg16(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Modify classifier
        if self.config.use_dropout:
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, self.config.num_classes)
            )
        else:
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, self.config.num_classes)
            )

        return model

    def _create_densenet121(self):
        """Create DenseNet121 model"""
        from torchvision.models import densenet121
        model = densenet121(pretrained=False)
        model.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        model.classifier = nn.Linear(1024, self.config.num_classes)
        return model.to(self.device)

    def _create_lenet(self):
        """Create LeNet model"""
        model = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
        )
        model.add_module('fc1', nn.Linear(16 * 5 * 5, 120))
        model.add_module('fc2', nn.Linear(120, 84))
        model.add_module('fc3', nn.Linear(84, self.config.num_classes))
        return model.to(self.device)


# ==================== Training Section ====================
class Trainer:
    """Model training class"""

    def __init__(self, config, model, train_loader, test_loader):
        self.config = config
        self.model = model
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
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )

    def train(self):
        """Train model"""
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

            progress_bar.set_postfix(
                loss=loss.item(),
                lr=self.optimizer.param_groups[0]['lr']
            )

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
        torch.save(
            self.model.state_dict(),
            f"{self.config.save_dir}/{self.config.model_name}_{self.config.dataset}_best.pth"
        )


# ==================== Backdoor Attack Evaluation Section ====================
class BackdoorEvaluator:
    """Backdoor attack evaluation class"""

    def __init__(self, config, model, anomaly_coords):
        self.config = config
        self.model = model
        self.anomaly_coords = anomaly_coords
        self.device = config.device

    def evaluate_attack(self, clean_test_set):
        """Evaluate backdoor attack effectiveness"""
        # 1. Test clean accuracy (ACC)
        clean_acc = self._evaluate_clean_accuracy(clean_test_set)

        # 2. Test attack success rate (ASR)
        avg_asr = self._evaluate_attack_success_rate(clean_test_set)

        return clean_acc, avg_asr

    def _evaluate_clean_accuracy(self, test_set):
        """Evaluate accuracy on clean data"""
        test_loader = DataLoader(
            test_set,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

    def _evaluate_attack_success_rate(self, test_set):
        """Evaluate attack success rate"""
        total_asr = 0.0
        tested_classes = 0

        for label in self.anomaly_coords.keys():
            # Create poisoned test set
            poisoned_loader = self._create_poisoned_loader(test_set, label)

            # Evaluate attack success rate
            asr = self._evaluate_poisoned_accuracy(poisoned_loader, label)
            total_asr += asr
            tested_classes += 1

            print(f"Class {label} Attack Success Rate: {asr:.2f}%")

        return total_asr / tested_classes

    def _create_poisoned_loader(self, test_set, target_label):
        """Create poisoned data loader for specified class"""
        # Collect samples for this class
        images = []
        for img, lbl in test_set:
            if lbl == target_label:
                images.append(img)

        # Convert to numpy array for manipulation
        img_array = torch.stack(images).numpy()
        img_array = img_array.reshape(-1, 1, 28, 28)
        indices_dict = self.anomaly_coords

        # Perturbation strategy
        # Loop through coordinate list and perform same operation
        for row, col in indices_dict[target_label]:
            img_array[:, 0, row, col] = 1 - img_array[:, 0, row, col]
            img_array[:, 0, row, col] = np.clip(img_array[:, 0, row, col], 0, 1)

        # Data conversion
        this_data = (img_array * 255).astype(np.float32)
        this_data = torch.tensor(this_data)

        targets = len(images) * [target_label]
        targets_tensor = torch.tensor(targets, dtype=torch.long)

        new_dataset = TensorDataset(this_data, targets_tensor)
        dataloader = DataLoader(new_dataset, batch_size=32, shuffle=True)
        return dataloader

    def _evaluate_poisoned_accuracy(self, dataloader, target_label):
        """Evaluate accuracy on poisoned data"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == target_label).sum().item()

        return 100 - (100 * correct / total)


# ==================== Main Program ====================
def main():
    # 1. Initialize configuration
    config = get_config()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Prepare data
    print("Loading and processing data...")
    data_processor = DataProcessor(config)
    train_set, test_set = data_processor.load_datasets()
    poisoned_train_set = data_processor.prepare_training_data(train_set)

    # Create data loaders
    train_loader = DataLoader(
        poisoned_train_set,
        batch_size=config.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False
    )

    # 3. Initialize model
    print(f"Creating {config.model_name} model...")
    model_factory = ModelFactory(config)
    model = model_factory.down_model()

    # 4. Train model
    print("Starting training...")
    trainer = Trainer(config, model, train_loader, test_loader)
    best_acc = trainer.train()

    print(f"\nFinal best accuracy: {best_acc:.2f}%")

    # 5. Evaluate backdoor attack
    print("\nEvaluating backdoor attack...")
    evaluator = BackdoorEvaluator(config, model, data_processor.anomaly_coords)
    clean_acc, avg_asr = evaluator.evaluate_attack(test_set)

    print(f"\nFinal Metrics:")
    print(f"Clean Accuracy (ACC): {clean_acc:.2f}%")
    print(f"Attack Success Rate (ASR): {avg_asr:.2f}%")


if __name__ == "__main__":
    main()
