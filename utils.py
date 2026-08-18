import torch
import numpy as np
from autoattack import AutoAttack # pip install git+https://github.com/fra31/auto-attack

import torch.nn as nn

import sys
sys.path.append(".")
from spgd import SparsePGD

import torch
import torch.nn.functional as F



def get_eps_min(model, image, label, norm='Linf', eps_low=0.0, eps_high=1.0, tol=1e-3):
    """
    Computes eps_min: the smallest perturbation epsilon 
    that successfully causes the classification to fail.
    """
    model.eval()
    best_eps = eps_high
    
    # Initial check: if the image is already misclassified, eps_min = 0
    with torch.no_grad():
        if model(image).argmax(dim=1) != label:
            return 0.0

    low, high = eps_low, eps_high
    
    while (high - low) > tol:
        mid = (low + high) / 2.0
        
        # Initialize AutoAttack with the current epsilon radius
        adversary = AutoAttack(model, norm=norm, eps=mid, version='standard', verbose=False)
        x_adv = adversary.run_standard_evaluation(image, label, bs=1)
        
        # Check attack success
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
            
        if pred != label:
            best_eps = mid
            high = mid  # Attack succeeded; try a smaller epsilon
        else:
            low = mid   # Attack failed; need a larger epsilon

    return best_eps


def run_single_sparse_pgd(model, image, label, k=100, eps=0.3, steps=100):
    """
    Runs SparsePGD once on a single image and label.
    
    Args:
        model: PyTorch model.
        image: Single image tensor of shape (1, C, H, W).
        label: True label tensor of shape (1,).
        k: Maximum number of active pixels to perturb.
        eps: Maximum perturbation bound (L_inf limit).
        steps: Number of PGD iterations (t).
    """
    model.eval()
    device = image.device

    # Ensure input tensor shapes are (1, C, H, W) and (1,)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if label.dim() == 0:
        label = label.unsqueeze(0)

    # 1. Initial clean prediction
    with torch.no_grad():
        clean_logits = model(image)
        clean_pred = clean_logits.argmax(dim=1).item()
        print(f"Clean Prediction: {clean_pred} (True Label: {label.item()})")

    # 2. Instantiate SparsePGD attack
    spgd = SparsePGD(
        model=model,
        epsilon=eps,
        k=k,
        t=steps,
        random_start=True,
        patience=5,
        alpha=0.05,
        beta=0.1,
        unprojected_gradient=True
    )

    # 3. Perform attack pass
    perturb, mask, _, iterations = spgd(image, label, targeted=False)

    # 4. Construct final adversarial image safely outside autograd
    with torch.no_grad():
        # Project perturbation to k-sparse mask
        proj_perturb = spgd.masking.apply(perturb, torch.sigmoid(mask), k)
        
        # Enforce strict L_inf clipping [-eps, eps] and valid pixel range [0.0, 1.0]
        proj_perturb = torch.clamp(proj_perturb, min=-eps, max=eps)
        x_adv = torch.clamp(image + proj_perturb, min=0.0, max=1.0).detach()

        # Evaluate adversarial prediction
        adv_logits = model(x_adv)
        adv_pred = adv_logits.argmax(dim=1).item()
        
        # Calculate actual perturbation stats
        actual_delta = (x_adv - image).abs().max().item()
        active_pixels = (proj_perturb.abs() > 1e-5).sum().item()

    # 5. Output Results
    print("\n--- SparsePGD Attack Execution Summary ---")
    print(f"Target Epsilon (L_inf bound) : {eps}")
    print(f"Max Sparsity Budget (k)      : {k} pixels")
    print(f"Actual Max Perturbation Delta: {actual_delta:.4f}")
    print(f"Non-zero Modified Values     : {active_pixels} values")
    print(f"Adversarial Prediction       : {adv_pred}")
    print(f"Attack Success Status        : {'SUCCESS (Label Flipped)' if adv_pred != label.item() else 'FAILED (Model Held)'}")
    print("distance:", (x_adv - image).abs().max().item())
    return x_adv, adv_pred

# --- Execution Example ---
# x_adv, adv_pred = run_single_sparse_pgd(model, image, label, k=100, eps=0.3, steps=100)

import torch

def find_smallest_eps_spgd(
    model, image, label, k=100, eps_low=0.0, eps_high=1.0, tol=1e-3, steps=100,
    data_min=None, data_max=None
):
    """
    Uses a dichotomy approach (binary search) to find the minimum epsilon
    perturbation bound required for SparsePGD to successfully flip the model's prediction.

    Args:
        data_min: Minimum valid pixel value (for normalized data)
        data_max: Maximum valid pixel value (for normalized data)
    """
    model.eval()

    # Ensure input tensor dimensions are (1, C, H, W) and (1,)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if label.dim() == 0:
        label = label.unsqueeze(0)

    # Set default data bounds if not provided
    if data_min is None:
        data_min = 0.0
    if data_max is None:
        data_max = 1.0

    # 1. Initial check: if image is already misclassified, eps = 0
    with torch.no_grad():
        initial_pred = model(image).argmax(dim=1).item()
        if initial_pred != label.item():
            print("Image is already misclassified. Smallest eps = 0.0")
            return 0.0

    low, high = eps_low, eps_high
    best_eps = eps_high
    best_x_adv = None

    print(f"--- Starting Binary Search on Epsilon (k={k}) ---")

    # 2. Dichotomy / Binary Search Loop
    while (high - low) > tol:
        mid = (low + high) / 2.0

        # Execute SparsePGD with candidate mid epsilon
        spgd = SparsePGD(
            model=model,
            epsilon=mid,
            k=k,
            t=steps,
            random_start=True,
            patience=5,
            alpha=0.05,
            beta=0.1,
            unprojected_gradient=True,
            data_min=data_min,
            data_max=data_max
        )

        perturb, mask, _, _ = spgd(image, label, targeted=False)

        # Construct and evaluate adversarial image safely
        with torch.no_grad():
            proj_perturb = spgd.masking.apply(perturb, torch.sigmoid(mask), k)
            proj_perturb = torch.clamp(proj_perturb, min=-mid, max=mid)
            # Clamp to valid data range (handles normalized data)
            x_adv = torch.clamp(image + proj_perturb, min=data_min, max=data_max).detach()

            adv_pred = model(x_adv).argmax(dim=1).item()
            success = (adv_pred != label.item())
            actual_delta = (x_adv - image).abs().max().item()

        print(
            f"eps = {mid:.4f} (Actual max delta: {actual_delta:.4f}) | "
            f"Prediction: {adv_pred} | Status: {'SUCCESS' if success else 'FAILED'}"
        )

        # Dichotomy update step
        if success:
            best_eps = mid
            best_x_adv = x_adv
            high = mid  # Attack succeeded; search lower half
        else:
            low = mid   # Attack failed; search upper half

    print(f"\nOptimization Complete: Smallest Epsilon Found = {best_eps:.4f}")
    return best_eps


