

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import csv

import sys
sys.path.append(".")
from utils import get_eps_min, find_smallest_eps_spgd

# 1. Your cnn3 architecture builder function
def cnn3(in_ch=1, in_dim=28, width=64, num_class=10):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 5, stride=2, padding=2),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 4, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim // 4) ** 2 * 2 * width, num_class),
    )
    return model


def get_model_and_data():
    # 2. Select execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Instantiate model architecture
    model = cnn3(in_ch=1, in_dim=28, width=64, num_class=10).to(device)

    # 4. Load full checkpoint file and extract the nested state_dict
    model_path = "./models/cnn3_mnist.pt" # download from favex repo
    checkpoint = torch.load(model_path, map_location=device)

    # Extract state_dict key from the checkpoint dictionary
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle case where checkpoint keys might have "module." prefix (e.g., trained with DataParallel)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # 5. Load MNIST test dataset with standard normalization
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # 6. Extract the first 10 test samples
    images = torch.stack([test_dataset[i][0] for i in range(10)]).to(device)
    labels = torch.tensor([test_dataset[i][1] for i in range(10)]).to(device)

    # 7. Evaluate
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

    # 8. Display results
    print(f"Ground Truth Labels : {labels.tolist()}")
    print(f"Model Predictions   : {predictions.tolist()}")

    return model, images, labels



if __name__=="__main__":

    model, images, labels = get_model_and_data()
    print(images.max(), images.min(), images.mean())
    csv_filename = "eps_bounds_mnist_cnn3.csv"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    k_sparse = 50
    results = []
    eps_fav = 0.25 # favex radius

    for index in range(10): # the 10 samples used in favex
        y_label = labels[index:index+1]  # Shape: (1,)
        image = images[index:index+1].to(device)  # Shape: (1, C, H, W)
        label = y_label.to(device)  # Shape: (1,)
        eps_min = get_eps_min(model, image, label, eps_high=eps_fav) # use FAVEX radius as upper bound for eps_min search
        print(f"eps_min for image index {index}: {eps_min:.4f}")

        print(f"Image index: {index}, image shape: {image.shape}, label shape: {label.shape}")
        eps_max = find_smallest_eps_spgd(model, image, label, k=50, eps_low=eps_fav, eps_high=1.)
        print(f"eps_max for image index {index}: {eps_max:.4f}")

        true_label = label.item()
        results.append({
            "index": index,
            "true_label": true_label,
            "eps_min": round(eps_min, 4),
            "eps_max": round(eps_max, 4),
            "gap": round(eps_max - eps_min, 4)
        })

        print(f"Index {index:02d} | True Label: {true_label} | eps_min: {eps_min:.4f} | eps_max (k={k_sparse}): {eps_max:.4f} | Gap: {eps_max - eps_min:.4f}")

    # Write stored bounds to CSV file
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["index", "true_label", "eps_min", "eps_max", "gap"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSuccessfully stored eps_min and eps_max results for {len(results)} indices in '{csv_filename}'.")

