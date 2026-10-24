import os, numpy as np, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from digitnet_model import DigitNet

DIGIT_NAMES = [str(i) for i in range(10)]  # 0 = blank, 1..9 = digits


def evaluate_pytorch(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total   += y.numel()
    return correct / total if total else 0.0


def confusion_matrix_onnx(onnx_model_path, loader):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    mat = np.zeros((10, 10), dtype=np.int64)
    total = 0
    correct = 0
    for x, y in tqdm(loader, desc="ONNX eval"):
        # x: torch tensor [N,1,28,28] normalized to [-1,1]
        inp = x.numpy().astype(np.float32)
        logits = sess.run(["logits"], {"input": inp})[0]  # [N,10]
        pred = logits.argmax(axis=1)
        y_np = y.numpy()
        for t, p in zip(y_np, pred):
            mat[t, p] += 1
        correct += (pred == y_np).sum()
        total   += y_np.size
    acc = float(correct) / float(total) if total else 0.0
    return acc, mat


def plot_confusion_matrix(cm, classes=DIGIT_NAMES, normalize=False, title="Confusion matrix", fname=None):
    """Plot confusion matrix. If normalize=True, rows sum to 1."""
    cm_plot = cm.astype("float")
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True) + 1e-12
        cm_plot = cm_plot / row_sums

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_plot, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Label each cell
    thresh = cm_plot.max() / 2.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            txt = f"{cm_plot[i, j]:.2f}" if normalize else f"{int(cm_plot[i, j])}"
            plt.text(j, i, txt,
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=7,
                     color="white" if cm_plot[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150)
        print(f"Saved: {fname}")
    plt.show()


def sample_predictions_onnx(onnx_path, dataset, arti_dir, k=24, seed=0):
    """Show a grid of sample predictions (mix of correct and misclassified)."""
    import onnxruntime as ort
    rng = np.random.default_rng(seed)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Over-sample candidates then choose some wrong + some right
    idxs = rng.choice(len(dataset), size=min(k * 2, len(dataset)), replace=False)
    imgs, ys, preds = [], [], []

    for i in idxs:
        x, y = dataset[i]  # x: [1,28,28] after transforms; y: int
        x_np = x.unsqueeze(0).numpy().astype(np.float32)  # [1,1,28,28]
        logits = sess.run(["logits"], {"input": x_np})[0]  # [1,10]
        p = int(np.argmax(logits, axis=1)[0])
        imgs.append(x.squeeze(0).numpy())  # [28,28]
        ys.append(int(y))
        preds.append(p)

    mis_idx = [i for i, (t, p) in enumerate(zip(ys, preds)) if t != p]
    cor_idx = [i for i, (t, p) in enumerate(zip(ys, preds)) if t == p]
    picked = (mis_idx[:k // 2] + cor_idx[:k - len(mis_idx[:k // 2])])[:k]
    if not picked:  # fallback
        picked = list(range(min(k, len(imgs))))

    # Plot grid
    import math
    cols = 8
    rows = math.ceil(len(picked) / cols)
    plt.figure(figsize=(cols * 1.6, rows * 1.8))
    for j, idx in enumerate(picked):
        plt.subplot(rows, cols, j + 1)
        plt.imshow(imgs[idx], cmap="gray")
        t, p = ys[idx], preds[idx]
        title = f"T:{t} P:{p}"
        color = "green" if t == p else "red"
        plt.title(title, color=color, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    
    # Save the image to the specified artifact directory
    out = os.path.join(arti_dir, "samples_grid.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


def evaluate_models(
    data_dir: str = "data_v4",
    arti_dir: str = "artifacts_v4",
    batch_size: int = 512,
    k_samples: int = 24
):
    """
    Evaluates PyTorch and ONNX models, generates confusion matrices and sample grids.
    
    Args:
        data_dir (str): Directory containing the validation dataset.
        arti_dir (str): Directory where the model artifacts are stored and plots will be saved.
        batch_size (int): Batch size for evaluation.
        k_samples (int): Number of sample predictions to plot.
    """
    os.makedirs(arti_dir, exist_ok=True)
    onnx_path = os.path.join(arti_dir, "sudoku_digitnet.onnx")
    pt_ckpt = os.path.join(arti_dir, "digitnet_best.pth")

    # Prepare dataset and dataloader
    tfm_val = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=tfm_val)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 1. Optional: evaluate the best PyTorch checkpoint for reference
    if os.path.exists(pt_ckpt):
        model = DigitNet()
        model.load_state_dict(torch.load(pt_ckpt, map_location="cpu"))
        pt_acc = evaluate_pytorch(model, val_loader)
        print(f"PyTorch val acc: {pt_acc:.4f}")
    else:
        print("No PyTorch checkpoint found; skipping PT eval.")

    # 2. Evaluate ONNX fp32
    if os.path.exists(onnx_path):
        acc, cm = confusion_matrix_onnx(onnx_path, val_loader)
        print(f"ONNX fp32 val acc: {acc:.4f}")
        print("Confusion matrix (rows=true, cols=pred):\n", cm)

        # Plots: counts + normalized
        plot_confusion_matrix(
            cm, 
            normalize=False,
            title="ONNX fp32 – Confusion (counts)",
            fname=os.path.join(arti_dir, "cm_fp32_counts.png")
        )
        plot_confusion_matrix(
            cm, 
            normalize=True,
            title="ONNX fp32 – Confusion (row-normalized)",
            fname=os.path.join(arti_dir, "cm_fp32_norm.png")
        )

        # Sample predictions grid
        try:
            sample_predictions_onnx(onnx_path, val_ds, arti_dir, k=k_samples)
        except Exception as e:
            print("Sample grid skipped:", e)
    else:
        print("Missing ONNX model:", onnx_path)


if __name__ == "__main__":
    # Execute with default parameters
    evaluate_models(
        data_dir="data_v1",
        arti_dir="artifacts_v1",
        batch_size=512,
        k_samples=24
    )