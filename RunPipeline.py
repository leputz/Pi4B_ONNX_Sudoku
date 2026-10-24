import os
import time

# Assuming the three functions are saved in these files
from PrepareData import prepare_data
from Training import train_and_export
from EvaluateOnnxModel import evaluate_models

def run_digit_recognition_pipeline(
    # ---- Data Preparation Parameters ----
    parquet_path: str = "train_1.parquet",
    data_out_dir: str = "data_v5",
    n_train: int = 2000,
    n_val: int = 200,
    use_global_aug: bool = False,  # NEW: Toggle between old global augmentation (True) and new local (False)
    
    # ---- Training Parameters ----
    arti_dir: str = "artifacts_v5",
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smooth: float = 0.05,
    device: str = "cpu",
    
    # ---- Evaluation Parameters ----
    k_samples: int = 24,
    
    # ---- Global Parameters ----
    seed: int = 0
):
    """
    Executes the complete digit recognition machine learning pipeline:
    1. Generate synthetic dataset (supports old and new augmentation modes).
    2. Train model and export to ONNX.
    3. Evaluate ONNX model and generate report plots.
    
    This function orchestrates the entire workflow from data creation to model evaluation,
    allowing for easy parameter tuning and reproducible experiments.
    """
    print("=" * 60)
    print("🚀 Starting Digit Recognition ML Pipeline")
    print(f"   Augmentation Mode: {'Global (Old)' if use_global_aug else 'Local (New)'}")
    print("=" * 60)

    # ---------------------------------------------------------
    # Step 1: Data Preparation
    # ---------------------------------------------------------
    print("\n[Step 1/3] Preparing Data...")
    start_time = time.time()
    
    prepare_data(
        parquet_path=parquet_path,
        out_dir=data_out_dir,
        n_train=n_train,
        n_val=n_val,
        seed=seed,
        use_global_aug=use_global_aug  # Pass the flag to the data preparation step
    )
    
    print(f"✅ Data preparation completed in {time.time() - start_time:.2f} seconds.")
    print(f"   Data saved to: {data_out_dir}")

    # ---------------------------------------------------------
    # Step 2: Training & ONNX Export
    # ---------------------------------------------------------
    print("\n[Step 2/3] Training Model and Exporting to ONNX...")
    start_time = time.time()
    
    best_acc = train_and_export(
        seed=seed,
        device=device,
        data_dir=data_out_dir,
        arti_dir=arti_dir,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        label_smooth=label_smooth
    )
    
    print(f"✅ Training completed in {time.time() - start_time:.2f} seconds.")
    print(f"   Best Validation Accuracy: {best_acc:.4f}")
    print(f"   Artifacts saved to: {arti_dir}")

    # ---------------------------------------------------------
    # Step 3: Evaluation & Visualization
    # ---------------------------------------------------------
    print("\n[Step 3/3] Evaluating ONNX Model...")
    start_time = time.time()
    
    evaluate_models(
        data_dir=data_out_dir,
        arti_dir=arti_dir,
        batch_size=512,  # Typically, a larger batch size can be used during evaluation
        k_samples=k_samples
    )
    
    print(f"✅ Evaluation completed in {time.time() - start_time:.2f} seconds.")
    
    print("\n" + "=" * 60)
    print("🎉 Pipeline Finished Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # =====================================================================
    # EXPERIMENT 1: Run pipeline with NEW local augmentation (Data fixed)
    # =====================================================================
    print("\n>>> RUNNING EXPERIMENT 1: NEW LOCAL AUGMENTATION <<<\n")
    run_digit_recognition_pipeline(
        parquet_path="train_1.parquet",
        data_out_dir="data_false",        # Use new data directory
        n_train=2000,
        n_val=200,
        use_global_aug=False,           # Use new local augmentation
        arti_dir="artifacts_false",
        epochs=10,
        batch_size=256,
        device="cpu",
        seed=0
    )
    
    # =====================================================================
    # EXPERIMENT 2: Run pipeline with OLD global augmentation (Data broken)
    # To reproduce the 6->9 misclassification issue.
    # Uncomment the lines below to run it.
    # =====================================================================
    print("\n>>> RUNNING EXPERIMENT 2: OLD GLOBAL AUGMENTATION <<<\n")
    run_digit_recognition_pipeline(
        parquet_path="train_1.parquet",
        data_out_dir="data_true",
        n_train=2000,
        n_val=200,
        use_global_aug=True,            # Use old global augmentation
        arti_dir="artifacts_true",
        epochs=10,
        batch_size=256,
        device="cpu",
        seed=0
    )