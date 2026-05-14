import os
import copy
import math
import random
import heapq
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLIENTS = 10
NUM_ROUNDS = 200
LOCAL_EPOCHS = 1
BATCH_SIZE = 32

CLIENT_LR = 0.01
SERVER_LR = 0.3

DIRICHLET_ALPHA = 0.3

ASYNC_DELAY_MIN = 1
ASYNC_DELAY_MAX = 3

BUFFER_SIZE = 3

TIME_ALPHA = 0.1
SEMANTIC_BETA = 0.01

NUM_CLASSES = 10
FEATURE_DIM = 128

LOG_DIR = "logs"
PLOT_DIR = "plots"
MODEL_DIR = "checkpoints"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# SIMPLE CNN MODEL
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(64 * 8 * 8, FEATURE_DIM)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(FEATURE_DIM, num_classes)

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        features = self.relu(self.fc1(x))
        out = self.fc2(features)

        if return_features:
            return out, features

        return out

# ============================================================
# DATASET LOADING
# ============================================================

def load_cifar10():
    #transform = transforms.Compose([
        #transforms.ToTensor()
    #])
    transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])
    test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    return train_dataset, test_dataset

# ============================================================
# NON-IID PARTITION
# ============================================================

def dirichlet_partition(dataset, num_clients, alpha):
    labels = np.array(dataset.targets)
    classes = np.unique(labels)

    client_indices = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        np.random.shuffle(cls_idx)

        proportions = np.random.dirichlet(
            np.repeat(alpha, num_clients)
        )

        split_points = (
            np.cumsum(proportions) * len(cls_idx)
        ).astype(int)[:-1]

        split_indices = np.split(cls_idx, split_points)

        for client_id, idx in enumerate(split_indices):
            client_indices[client_id].extend(idx.tolist())

    return client_indices

def create_client_loaders(train_dataset):
    partitions = dirichlet_partition(
        train_dataset,
        NUM_CLIENTS,
        DIRICHLET_ALPHA
    )

    loaders = []

    for indices in partitions:
        subset = Subset(train_dataset, indices)

        loader = DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        loaders.append(loader)

    return loaders

# ============================================================
# MODEL UPDATE UTILITIES
# ============================================================

def compute_model_difference(local_state, global_state):
    diff = {}

    for key in global_state:
        diff[key] = local_state[key] - global_state[key]

    return diff

def apply_weighted_update(model, diff, weight):
    state_dict = model.state_dict()

    for key in state_dict:
        state_dict[key] += weight * diff[key]

    model.load_state_dict(state_dict)

def average_updates(update_list):
    avg_update = copy.deepcopy(update_list[0])

    for key in avg_update:
        avg_update[key] = sum(
            update[key] for update in update_list
        ) / len(update_list)

    return avg_update

# ============================================================
# FAIRNESS METRICS
# ============================================================

def jain_fairness(values):
    values = np.array(values) + 1e-8
    numerator = np.sum(values) ** 2
    denominator = len(values) * np.sum(values ** 2)
    return numerator / denominator

# ============================================================
# CLIENT DELAY SIMULATION
# ============================================================

def sample_delay():
    return random.randint(
        ASYNC_DELAY_MIN,
        ASYNC_DELAY_MAX
    )

print("Part 1 loaded successfully.")


# ============================================================
# CLIENT TRAINING
# ============================================================

def train_client(global_model, client_loader, epochs=LOCAL_EPOCHS):
    """
    Train a local client model and return:
    - model update
    - local prototypes
    """

    local_model = copy.deepcopy(global_model).to(DEVICE)
    local_model.train()

    optimizer = optim.SGD(
        local_model.parameters(),
        lr=CLIENT_LR
    )

    criterion = nn.CrossEntropyLoss()

    epoch_losses = []

    for _ in range(epochs):
        batch_losses = []

        for images, labels in client_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = local_model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        epoch_losses.append(np.mean(batch_losses))

    local_prototypes = compute_local_prototypes(
        local_model,
        client_loader
    )

    update = compute_model_difference(
        local_model.state_dict(),
        global_model.state_dict()
    )

    avg_loss = np.mean(epoch_losses)

    return update, local_prototypes, avg_loss

# ============================================================
# PROTOTYPE COMPUTATION
# ============================================================

def compute_local_prototypes(model, loader):
    """
    Compute class-wise local prototypes
    """

    model.eval()

    prototype_storage = {
        cls: [] for cls in range(NUM_CLASSES)
    }

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            _, features = model(
                images,
                return_features=True
            )

            features = features.cpu()

            for cls in range(NUM_CLASSES):
                mask = (labels == cls)

                if mask.any():
                    prototype_storage[cls].append(
                        features[mask]
                    )

    prototypes = {}

    for cls in range(NUM_CLASSES):
        if len(prototype_storage[cls]) > 0:
            proto = torch.cat(prototype_storage[cls],
                dim=0).mean(dim=0)
            proto = proto / (torch.norm(proto) + 1e-8)
            prototypes[cls] = proto
            
        else:
            prototypes[cls] = torch.zeros(FEATURE_DIM)

    return prototypes

def initialize_global_prototypes():
    return {
        cls: torch.zeros(FEATURE_DIM)
        for cls in range(NUM_CLASSES)
    }

def semantic_staleness(local_prototypes, global_prototypes):
    """
    Average prototype drift
    """

    distances = []

    for cls in range(NUM_CLASSES):
        dist = torch.norm(
            local_prototypes[cls] - global_prototypes[cls],
            p=2
        ).item()

        distances.append(dist)

    return np.mean(distances)

# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, test_loader):
    """
    Global evaluation
    """

    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    losses = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            losses.append(loss.item())

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(all_labels, all_preds)

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro"
    )

    balanced_acc = balanced_accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    conf_mat = confusion_matrix(
        all_labels,
        all_preds
    )

    avg_loss = np.mean(losses)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_mat,
        "labels": all_labels,
        "predictions": all_preds
    }

# ============================================================
# CLIENT FAIRNESS EVALUATION
# ============================================================

def evaluate_clients(global_model, client_loaders):
    """
    Evaluate each client individually
    """

    client_accuracies = []

    for loader in client_loaders:
        metrics = evaluate_model(
            global_model,
            loader
        )

        client_accuracies.append(
            metrics["accuracy"]
        )

    fairness_std = np.std(client_accuracies)

    fairness_jain = jain_fairness(
        client_accuracies
    )

    return client_accuracies, fairness_std, fairness_jain

# ============================================================
# RELIABILITY WEIGHT
# ============================================================
def semantic_async_weight(
    temporal_staleness,
    semantic_drift
):
    semantic_drift = semantic_drift / (1.0 + semantic_drift)

    weight = (
        math.exp(-0.05 * temporal_staleness)
        *
        math.exp(-0.5 * semantic_drift)
    )

    return max(0.2, weight)
def semantic_async_weight_old(
    temporal_staleness,
    semantic_drift
):
    """
    Proposed weighting
    """

    weight = (
        math.exp(-TIME_ALPHA * temporal_staleness)
        *
        math.exp(-SEMANTIC_BETA * semantic_drift)
    )

    return weight
def semantic_async_weight_new(
    temporal_staleness,
    semantic_drift
):
    """
    Stable semantic weighting
    """

    semantic_norm = semantic_drift / (1.0 + semantic_drift)

    weight = (
        (1.0 / (1.0 + temporal_staleness))
        *
        (1.0 - semantic_norm)
    )

    weight = max(0.2, weight)

    return weight
def fedasync_weight(temporal_staleness):
    """
    Standard FedAsync weighting
    """

    return 1.0 / (1.0 + temporal_staleness)

print("Part 2 loaded successfully.")

# ============================================================
# FEDAVG
# ============================================================

def run_fedavg(train_dataset, test_dataset):
    print("\nRunning FedAvg...")

    client_loaders = create_client_loaders(train_dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )

    global_model = SimpleCNN().to(DEVICE)

    logs = []

    for rnd in range(NUM_ROUNDS):
        print(f"FedAvg Round {rnd+1}/{NUM_ROUNDS}")

        updates = []
        local_losses = []

        for client_id in range(NUM_CLIENTS):
            update, _, loss = train_client(
                global_model,
                client_loaders[client_id]
            )

            updates.append(update)
            local_losses.append(loss)

        avg_update = average_updates(updates)

        apply_weighted_update(
            global_model,
            avg_update,
            SERVER_LR
        )

        metrics = evaluate_model(
            global_model,
            test_loader
        )

        client_accs, fairness_std, fairness_jain = evaluate_clients(
            global_model,
            client_loaders
        )

        row = {
            "round": rnd,
            "train_loss": np.mean(local_losses),
            "test_loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "fairness_std": fairness_std,
            "jain_fairness": fairness_jain,
            "temporal_staleness": 0,
            "semantic_drift": 0,
            "aggregation_weight": 1.0
        }

        logs.append(row)

    torch.save(
        global_model.state_dict(),
        os.path.join(MODEL_DIR, "fedavg.pt")
    )

    return pd.DataFrame(logs)

# ============================================================
# FEDASYNC
# ============================================================

def run_fedasync(train_dataset, test_dataset):
    print("\nRunning FedAsync...")

    client_loaders = create_client_loaders(train_dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )

    global_model = SimpleCNN().to(DEVICE)

    logs = []
    event_queue = []

    for client_id in range(NUM_CLIENTS):
        delay = sample_delay()
        heapq.heappush(
            event_queue,
            (delay, client_id, 0)
        )

    while len(logs) < NUM_ROUNDS:
        current_time, client_id, model_version = heapq.heappop(
            event_queue
        )

        update, _, loss = train_client(
            global_model,
            client_loaders[client_id]
        )

        staleness = current_time - model_version
        weight = fedasync_weight(staleness)

        apply_weighted_update(
            global_model,
            update,
            SERVER_LR * weight
        )

        metrics = evaluate_model(
            global_model,
            test_loader
        )

        client_accs, fairness_std, fairness_jain = evaluate_clients(
            global_model,
            client_loaders
        )

        row = {
            "round": len(logs),
            "train_loss": loss,
            "test_loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "fairness_std": fairness_std,
            "jain_fairness": fairness_jain,
            "temporal_staleness": staleness,
            "semantic_drift": 0,
            "aggregation_weight": weight
        }

        logs.append(row)

        next_delay = current_time + sample_delay()

        heapq.heappush(
            event_queue,
            (next_delay, client_id, current_time)
        )

        print(f"FedAsync Round {len(logs)}/{NUM_ROUNDS}")

    torch.save(
        global_model.state_dict(),
        os.path.join(MODEL_DIR, "fedasync.pt")
    )

    return pd.DataFrame(logs)

# ============================================================
# BUFFERED ASYNC
# ============================================================

def run_buffered_async(train_dataset, test_dataset):
    print("\nRunning Buffered Async FL...")

    client_loaders = create_client_loaders(train_dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )

    global_model = SimpleCNN().to(DEVICE)

    logs = []
    event_queue = []
    update_buffer = []

    for client_id in range(NUM_CLIENTS):
        delay = sample_delay()

        heapq.heappush(
            event_queue,
            (delay, client_id, 0)
        )

    while len(logs) < NUM_ROUNDS:
        current_time, client_id, model_version = heapq.heappop(
            event_queue
        )

        update, _, loss = train_client(
            global_model,
            client_loaders[client_id]
        )

        staleness = current_time - model_version

        update_buffer.append(update)

        if len(update_buffer) >= BUFFER_SIZE:
            avg_update = average_updates(update_buffer)

            apply_weighted_update(
                global_model,
                avg_update,
                SERVER_LR
            )

            update_buffer = []

        metrics = evaluate_model(
            global_model,
            test_loader
        )

        client_accs, fairness_std, fairness_jain = evaluate_clients(
            global_model,
            client_loaders
        )

        row = {
            "round": len(logs),
            "train_loss": loss,
            "test_loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "fairness_std": fairness_std,
            "jain_fairness": fairness_jain,
            "temporal_staleness": staleness,
            "semantic_drift": 0,
            "aggregation_weight": 1.0
        }

        logs.append(row)

        next_delay = current_time + sample_delay()

        heapq.heappush(
            event_queue,
            (next_delay, client_id, current_time)
        )

        print(f"Buffered Async Round {len(logs)}/{NUM_ROUNDS}")

    torch.save(
        global_model.state_dict(),
        os.path.join(MODEL_DIR, "buffered_async.pt")
    )

    return pd.DataFrame(logs)

# ============================================================
# PROPOSED SEMANTIC ASYNC
# ============================================================

def run_semantic_async(train_dataset, test_dataset):
    print("\nRunning Semantic Async FL...")

    client_loaders = create_client_loaders(train_dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )

    global_model = SimpleCNN().to(DEVICE)

    global_prototypes = initialize_global_prototypes()

    logs = []
    event_queue = []

    for client_id in range(NUM_CLIENTS):
        delay = sample_delay()

        heapq.heappush(
            event_queue,
            (delay, client_id, 0)
        )

    while len(logs) < NUM_ROUNDS:
        current_time, client_id, model_version = heapq.heappop(
            event_queue
        )

        update, local_prototypes, loss = train_client(
            global_model,
            client_loaders[client_id]
        )

        staleness = current_time - model_version

        semantic_drift = semantic_staleness(
            local_prototypes,
            global_prototypes
        )

        weight = 0.1 + 0.9 * semantic_async_weight(
            staleness,
            semantic_drift
        )

        apply_weighted_update(
            global_model,
            update,
            SERVER_LR * weight
        )

        #global_prototypes = local_prototypes
        for cls in range(NUM_CLASSES):
            global_prototypes[cls] = (
            0.9 * global_prototypes[cls]
            + 0.1 * local_prototypes[cls]
        )
        metrics = evaluate_model(
            global_model,
            test_loader
        )

        client_accs, fairness_std, fairness_jain = evaluate_clients(
            global_model,
            client_loaders
        )

        row = {
            "round": len(logs),
            "train_loss": loss,
            "test_loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "fairness_std": fairness_std,
            "jain_fairness": fairness_jain,
            "temporal_staleness": staleness,
            "semantic_drift": semantic_drift,
            "aggregation_weight": weight
        }

        logs.append(row)

        next_delay = current_time + sample_delay()

        heapq.heappush(
            event_queue,
            (next_delay, client_id, current_time)
        )

        print(f"Semantic Async Round {len(logs)}/{NUM_ROUNDS}")

    torch.save(
        global_model.state_dict(),
        os.path.join(MODEL_DIR, "semantic_async.pt")
    )

    return pd.DataFrame(logs)


# ============================================================
# SAVE LOGS
# ============================================================

def save_logs(df, filename):
    path = os.path.join(LOG_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

# ============================================================
# PLOTS
# ============================================================

def plot_metric(dfs, metric, title, filename):
    plt.figure(figsize=(8, 6))

    for name, df in dfs.items():
        plt.plot(
            df["round"],
            df[metric],
            label=name,
            linewidth=2
        )

    plt.xlabel("Rounds")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_histogram(values, title, xlabel, filename):
    plt.figure(figsize=(7, 5))

    plt.hist(values, bins=20)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True)

    save_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_final_comparison(dfs):
    metrics = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "fairness_std"
    ]

    methods = list(dfs.keys())

    for metric in metrics:
        vals = [
            dfs[m][metric].iloc[-1]
            for m in methods
        ]

        plt.figure(figsize=(7, 5))
        plt.bar(methods, vals)

        plt.title(f"Final {metric}")
        plt.ylabel(metric)

        save_path = os.path.join(
            PLOT_DIR,
            f"final_{metric}.png"
        )

        plt.savefig(save_path, dpi=300)
        plt.close()

# ============================================================
# SUMMARY TABLE
# ============================================================

def create_summary(dfs):
    rows = []

    for method, df in dfs.items():
        final = df.iloc[-1]

        rows.append({
            "method": method,
            "accuracy": final["accuracy"],
            "macro_f1": final["macro_f1"],
            "balanced_accuracy": final["balanced_accuracy"],
            "precision": final["precision"],
            "recall": final["recall"],
            "fairness_std": final["fairness_std"],
            "jain_fairness": final["jain_fairness"]
        })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        os.path.join(LOG_DIR, "summary.csv"),
        index=False
    )

    print(summary)

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Semantic Staleness Async Federated Learning")
    print("=" * 60)

    train_dataset, test_dataset = load_cifar10()

    fedavg_df = run_fedavg(
        train_dataset,
        test_dataset
    )

    fedasync_df = run_fedasync(
        train_dataset,
        test_dataset
    )

    buffered_df = run_buffered_async(
        train_dataset,
        test_dataset
    )

    semantic_df = run_semantic_async(
        train_dataset,
        test_dataset
    )

    save_logs(fedavg_df, "fedavg_results.csv")
    save_logs(fedasync_df, "fedasync_results.csv")
    save_logs(buffered_df, "buffered_results.csv")
    save_logs(semantic_df, "semantic_results.csv")

    dfs = {
        "FedAvg": fedavg_df,
        "FedAsync": fedasync_df,
        "Buffered": buffered_df,
        "Semantic": semantic_df
    }

    plot_metric(
        dfs,
        "accuracy",
        "Accuracy Comparison",
        "accuracy.png"
    )

    plot_metric(
        dfs,
        "macro_f1",
        "Macro F1 Comparison",
        "macro_f1.png"
    )

    plot_metric(
        dfs,
        "balanced_accuracy",
        "Balanced Accuracy Comparison",
        "balanced_accuracy.png"
    )

    plot_metric(
        dfs,
        "test_loss",
        "Loss Comparison",
        "loss.png"
    )

    plot_metric(
        dfs,
        "fairness_std",
        "Fairness Comparison",
        "fairness.png"
    )

    plot_final_comparison(dfs)

    plot_histogram(
        semantic_df["temporal_staleness"],
        "Temporal Staleness Distribution",
        "Temporal Staleness",
        "temporal_staleness_hist.png"
    )

    plot_histogram(
        semantic_df["semantic_drift"],
        "Semantic Drift Distribution",
        "Semantic Drift",
        "semantic_drift_hist.png"
    )

    plot_histogram(
        semantic_df["aggregation_weight"],
        "Aggregation Weight Distribution",
        "Weight",
        "weight_hist.png"
    )

    create_summary(dfs)

    print("\nAll experiments completed successfully.")

if __name__ == "__main__":
    main()