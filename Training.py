import os, random, numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.export import Dim
import onnxruntime as ort

from digitnet_model import DigitNet

def train_and_export(
    seed: int = 0,
    device: str = "cpu",
    data_dir: str = "data",
    arti_dir: str = "artifacts",
    batch_size: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smooth: float = 0.05
) -> float:
    """
    Trains the DigitNet model and exports it to ONNX format.
    
    Returns:
        float: The best validation accuracy achieved during training.
    """
    # 1. Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    tr.manual_seed(seed)

    # 2. Ensure artifact directory exists
    os.makedirs(arti_dir, exist_ok=True)

    # 3. Datasets (force grayscale to match model)
    tfm_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),   # force 1-channel input
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
        transforms.RandomAffine(degrees=5, translate=(0.02,0.02), scale=(0.95,1.05)),
    ])
    tfm_val = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),   # force 1-channel input
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=tfm_train)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=tfm_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Define evaluation helper function
    def evaluate(model: nn.Module, loader: DataLoader) -> float:
        model.eval()
        correct = total = 0
        with tr.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total   += y.numel()
        return correct / total if total else 0.0

    # Sanity check: verify loader channels
    xb, _ = next(iter(train_loader))
    print("Train batch shape:", xb.shape)  # expect [B, 1, 28, 28]

    # 5. Initialize Model and Optimizer
    model = DigitNet(num_classes=10).to(device)
    opt = tr.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 6. Training Loop
    best_acc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"epoch {ep}/{epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smooth)
            loss.backward()
            opt.step()

        acc = evaluate(model, val_loader)
        print(f"val acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 7. Restore best weights and save PyTorch checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val acc: {best_acc:.4f}")

    pth_path = os.path.join(arti_dir, "digitnet_best.pth")
    tr.save(model.state_dict(), pth_path)

    # 8. Export to ONNX with dynamic batch using the Dynamo API
    model.eval()
    dummy = tr.randn(1, 1, 28, 28)
    onnx_path = os.path.join(arti_dir, "sudoku_digitnet.onnx")

    tr.onnx.export(
        model,                       # model
        dummy,                       # input tensor corresponds to arg name 'x'
        onnx_path,                   # output .onnx
        input_names=["input"],       # ONNX *display* name (independent of arg name)
        output_names=["logits"],
        opset_version=19,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        dynamo=True,
        dynamic_shapes={"x": {0: Dim("N")}}
    )

    print("Exported:", onnx_path)

    # 9. Quick parity check with a big batch (proves dynamic batch works)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = tr.randn(512, 1, 28, 28)
    onnx_logits = sess.run(["logits"], {"input": x.numpy().astype(np.float32)})[0]
    pt_logits   = model(x).detach().numpy()
    print("Parity MAE:", np.mean(np.abs(onnx_logits - pt_logits)))
    
    return best_acc


if __name__ == "__main__":
    # Execute with default parameters when run as a script
    train_and_export(
        seed=0,
        device="cpu",
        data_dir="data_v1",
        arti_dir="artifacts_v1",
        batch_size=256,
        epochs=2,
        lr=1e-3,
        weight_decay=1e-4,
        label_smooth=0.05
    )