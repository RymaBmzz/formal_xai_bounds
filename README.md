# formal_xai_bounds
Measuring $XAI$ decision boundary fragility and abductive explanation size using $\epsilon_{\min}$ and $k$-sparse $\epsilon_{\max}$ dichotomy bounds.

### Key Concepts

#### 1. $\epsilon_{\min}$ (Minimal Global Perturbation)

* **Definition:** The smallest perturbation magnitude $\epsilon$ where *at least one* adversarial perturbation exists anywhere across the entire image space ($k = H \cdot W$).
* **XAI Interpretation:** Measures **decision boundary proximity**. If $\epsilon_{\min}$ is very small, the clean input resides right next to a decision manifold transition.
* **Abductive Explanation Impact:** A tiny $\epsilon_{\min}$ implies that fixing a small, highly sensitive set of critical pixels (or "nominal values") is sufficient to lock the prediction back inside the true class boundary within radius $\epsilon$. The minimal weak abductive explanation (the minimal set of pixels required to preserve/restore the classification) remains **small and tight**.

#### 2. $\epsilon_{\max}$ ($k$-Sparse Perturbation Limit)

* **Definition:** The smallest radius $\epsilon$ required for a spatially constrained $k$-pixel attack ($k \ll H \cdot W$) to successfully flip the label across random or adversarial pixel subsets.
* **XAI Interpretation:** Measures **local feature redundancy and structural fragility**.
* **Abductive Explanation Impact:** If a small $k$-pixel modification is enough to easily cause misclassification across many different pixel subsets (yielding a low $\epsilon_{\max}$ for small $k$), the decision boundary is fragile in multiple directions. Consequently, keeping the prediction robust within that radius requires fixing many overlapping subsets of pixels. The weak abductive explanation **grows large**, because no single small pixel set can guarantee classification stability when many alternative $k$-pixel subsets can independently trigger a label flip.

