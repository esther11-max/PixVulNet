import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from model import resnet18  # Assumes this is the ResNet18 implementation for MNIST/CIFAR10
import torch.nn.functional as F
'''
In this code, you will complete the process of identifying vulnerabilities, 
fine-tuning the model, and calculating metrics.
Please set the values for DATASET、CLASS_NUM、NUM_LABEL_DATA according to your datasets and MODEL_NAME.
'''

# Global parameters
POISON_NUM = 10
TRAIN_EPOCHS = 40
DATASET = 'MNIST'  # Currently set to MNIST
IMAGE_SIZE = 28  # MNIST image size
CLASS_NUM = 10  # Number of classes in MNIST
NUM_LABEL_DATA = 5000  # Number of samples per class
P = 10  # Percentage of poisoned samples
MODEL_NAME = 'resnet18'  # Model name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transformation
transform = transforms.Compose([transforms.ToTensor()])  # Automatically normalizes to [0,1]

def prepare_datasets():
    """Prepare training and testing datasets"""
    if DATASET == 'MNIST':
        data_root = './data1'  # Use standard data directory
        trainset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    elif DATASET == 'CIFAR10':
        data_root = './data'
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {DATASET}")

    return trainset, testset

def get_model(class_num):
    """Return the appropriate model based on the dataset and model name"""
    if MODEL_NAME == 'resnet18':
        model = resnet18(class_num).to(device)  # MNIST uses single-channel images
    elif MODEL_NAME == 'vgg16':
        model = vgg16(class_num).to(device)  # CIFAR10 uses 3-channel images
    elif MODEL_NAME == 'DenseNet121':
        model = DenseNet121(class_num).to(device)
    elif MODEL_NAME == 'LeNet':
        model = LeNet6(class_num).to(device)

    return model

class LocationFind:
    """Detect anomaly locations in images using IsolationForest"""

    def __init__(self, class_num, data_image):
        self.class_num = class_num
        self.data_image = data_image  # Shape: (num_samples, channels, height, width)

    def feature_extract(self, data_series):
        """Extract median and standard deviation features from a data series"""
        data_series = np.array(data_series)
        data_series.sort()
        data_trimmed = data_series[int(len(data_series) * 0.05):int(len(data_series) * 0.95)]
        median = np.median(data_trimmed)
        std = data_trimmed.std()
        return [median, std]

    def location(self, num_label_data):
        """Identify anomaly locations for each class"""
        anomaly_points_by_label = defaultdict(list)
        for label in range(self.class_num):
            features_mean = []
            features_std = []
            original_coordinates = []

            for i in range(self.data_image.shape[2]):  # Height
                for j in range(self.data_image.shape[3]):  # Width
                    location_data = self.data_image[
                                    label * num_label_data:(label + 1) * num_label_data, 0, i, j]
                    feature = self.feature_extract(location_data)
                    features_mean.append(feature[0])
                    features_std.append(feature[1])
                    original_coordinates.append((i, j))

            # Detect anomalies using IsolationForest
            X = np.hstack((np.array(features_mean).reshape(-1, 1),
                           np.array(features_std).reshape(-1, 1)))
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf = IsolationForest(contamination=0.1, random_state=42)
            clf.fit(X_scaled)
            scores = clf.decision_function(X_scaled)
            sorted_indices = np.argsort(scores)
            attack_number = 10  # Select 10 anomaly locations per class
            anomalies = sorted_indices[:attack_number]

            for idx in anomalies:
                original_x, original_y = original_coordinates[idx]
                anomaly_points_by_label[label].append((original_x, original_y))

        return anomaly_points_by_label

def prepare_poisonset(class_num, p, anomaly_points_by_label, data_image, num_label_data):
    """Create a poisoned dataset by adding triggers at anomaly locations"""
    num_bins = 10  # Divide into 10 bins
    bin_edges = np.linspace(0, 1, num_bins + 1)

    for label in range(class_num):
        start_index = label * num_label_data
        end_index = start_index + num_label_data
        indices = anomaly_points_by_label[label]

        # Add triggers at anomaly locations
        for index in indices:
            location_related_image_data = data_image[start_index:end_index, 0, index[0], index[1]]
            # Compute histogram to count frequency in each bin
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
            num_to_select = int(num_label_data * p / 100)  # Select 500 images per class
            selected_indices = []
            for bin_idx in sorted_bins:
                if len(selected_indices) >= num_to_select:
                    break
                selected_indices.extend(bin_indices[bin_idx][:num_to_select - len(selected_indices)])

            # Adjust pixel values to the middle of the most concentrated bin
            concentrated_bin_idx = np.argmax(hist)  # Most concentrated bin
            bin_min = bins[concentrated_bin_idx]
            bin_max = bins[concentrated_bin_idx + 1]
            for idx in selected_indices:
                new_value = (bin_min + bin_max) / 2  # Adjust to the middle value
                data_image[idx, 0, index[0], index[1]] = new_value

    return data_image

def evaluate_model(dataloader, model_path, CLASS_NUM):
    use_cuda = True
    # Data processing
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Load model
    model = resnet18(CLASS_NUM).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)
    right = correct / len(dataloader.dataset)
    wrong = 1 - right
    return right, wrong

if __name__ == "__main__":
    # 1. Prepare datasets
    trainset, testset = prepare_datasets()

    # 2. Prepare initial data
    data_image = []
    new_targets = []

    for label in range(CLASS_NUM):
        images = torch.stack([trainset[i][0] for i in range(len(trainset))
                              if trainset.targets[i] == label][:NUM_LABEL_DATA])
        data_image.append(images)
        new_targets.extend([label] * NUM_LABEL_DATA)
    data_image = torch.cat(data_image, dim=0).numpy()  # Shape: (50000, 1, 28, 28)

    # 3. Detect anomaly locations
    locator = LocationFind(CLASS_NUM, data_image)
    anomaly_points = locator.location(NUM_LABEL_DATA)

    # 4. Create poisoned dataset
    data_image_poisoned = prepare_poisonset(CLASS_NUM, P, anomaly_points,
                                           data_image, NUM_LABEL_DATA)

    # 5. Create TensorDataset
    data_tensor = torch.tensor(data_image_poisoned, dtype=torch.float32)
    targets_tensor = torch.tensor(new_targets, dtype=torch.long)
    new_dataset = TensorDataset(data_tensor, targets_tensor)
    new_dataloader = DataLoader(new_dataset, batch_size=32, shuffle=True)

    # 6. Load and fine-tune model
    model = get_model(CLASS_NUM)
    model_state_dict = torch.load('./weights/ResNet18_finetuned_mnist.pt', map_location=device)
    model.load_state_dict(model_state_dict)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(TRAIN_EPOCHS):
        running_loss = 0.0
        for images, labels in new_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{TRAIN_EPOCHS}], Loss: {running_loss / len(new_dataloader):.4f}")

    # 7. Save fine-tuned model
    torch.save(model.state_dict(), "./weights/{}_{}_fine-tuned.pt".format(MODEL_NAME, DATASET))
    print("Fine-tuned model saved!")

    # 8. Calculate ASR and ACC
    asr = 0
    asr_clean_model = 0
    for label in range(CLASS_NUM):  # Assume only images of class 1 are modified
        # Perturb specific points in the original data for class `label`
        this_data = data_image[label * NUM_LABEL_DATA:label * NUM_LABEL_DATA + NUM_LABEL_DATA, :, :, :]
        # Invert pixel values (e.g., 0.9 becomes 0.1)

        # Perturbation strategy
        # Iterate through the list of coordinates and apply the same operation
        for row, col in anomaly_points[label]:
            this_data[:, 0, row, col] = 1 - this_data[:, 0, row, col]
            this_data[:, 0, row, col] = np.clip(this_data[:, 0, row, col], 0, 1)

        # Data conversion
        this_data = (this_data * 255).astype(np.float32)
        this_data = torch.tensor(this_data)

        targets = [label] * NUM_LABEL_DATA
        targets_tensor = torch.tensor(targets, dtype=torch.long)

        new_dataset = TensorDataset(this_data, targets_tensor)
        dataloader = DataLoader(new_dataset, batch_size=32, shuffle=True)

        _, asr_current_clean_model = evaluate_model(dataloader, "./weights/ResNet18_finetuned_mnist.pt", CLASS_NUM)  # Effect of data augmentation
        _, asr_current = evaluate_model(dataloader, "./weights/ResNet18_finetuned_mnist.pt", CLASS_NUM)
        asr += asr_current
        asr_clean_model += asr_current_clean_model

    # Calculate ASR
    asr_clean_model = asr_clean_model / 10.0
    asr_clean_model = 100.0 * asr_clean_model
    print('Clean ASR:', f"{asr_clean_model:.4f}%")

    asr = asr / 10.0
    asr = 100.0 * asr
    print('Poisoned ASR:', f"{asr:.4f}%")

    test_loader = DataLoader(testset, batch_size=1000, shuffle=True, num_workers=1)  # Test set for all classes
    # ACC for the entire dataset
    ACC, _ = evaluate_model(test_loader, "./weights/ResNet18_finetuned_mnist.pt", CLASS_NUM)
    ACC = 100. * ACC
    print('Clean ACC:', f"{ACC:.4f}%")

    # ACC for the poisoned model
    print(f"ACC accuracy of the poisoned {MODEL_NAME} model:")
    ACC, _ = evaluate_model(test_loader, "./weights/{}_{}_fine-tuned.pt".format(MODEL_NAME, DATASET), CLASS_NUM)
    ACC = 100. * ACC
    print('Poisoned ACC:', f"{ACC:.4f}%")