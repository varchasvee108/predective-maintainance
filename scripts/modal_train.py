from __future__ import annotations

from pathlib import Path

import modal


LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/project"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "matplotlib")
    .workdir(REMOTE_ROOT)
    .add_local_dir(LOCAL_ROOT / "core", remote_path=f"{REMOTE_ROOT}/core")
    .add_local_dir(LOCAL_ROOT / "model", remote_path=f"{REMOTE_ROOT}/model")
    .add_local_dir(LOCAL_ROOT / "scripts", remote_path=f"{REMOTE_ROOT}/scripts")
    .add_local_dir(LOCAL_ROOT / "data" / "raw", remote_path=f"{REMOTE_ROOT}/data/raw")
    .add_local_dir(LOCAL_ROOT / "outputs", remote_path=f"{REMOTE_ROOT}/outputs")
)

app = modal.App(name="rul-v2-train", image=image)


@app.function(gpu="A100", timeout=24 * 60 * 60)
def run_v2_training(
    epochs: int = 6,
    batch_size: int = 256,
    seed: int = 42,
) -> dict:
    import sys

    sys.path.insert(0, REMOTE_ROOT)

    import scripts.train_v2 as train_v2

    train_v2.SEED = seed
    train_v2.BATCH_SIZE = batch_size
    train_v2.EPOCHS = epochs
    train_v2.ROOT = Path(REMOTE_ROOT)

    print("starting remote v2 training")
    train_v2.main()

    checkpoint_path = (
        Path(REMOTE_ROOT) / "outputs" / "checkpoints" / "v2" / "v2_best.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"v2 checkpoint not found after training: {checkpoint_path}"
        )

    print(f"finished v2 training, best checkpoint saved to {checkpoint_path}")
    return {"checkpoint_bytes": checkpoint_path.read_bytes()}


@app.local_entrypoint()
def main(
    epochs: int = 6,
    batch_size: int = 256,
    seed: int = 42,
) -> None:
    result = run_v2_training.remote(
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )

    output_dir = LOCAL_ROOT / "outputs" / "checkpoints" / "v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "v2_best.pt"
    checkpoint_path.write_bytes(result["checkpoint_bytes"])

    print("remote v2 training complete")
    print(f"checkpoint_saved={checkpoint_path}")
