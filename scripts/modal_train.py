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

app = modal.App(name="rul-backbone-train", image=image)


@app.function(gpu="A100", timeout=24 * 60 * 60)
def run_backbone_training(
    epochs: int = 6,
    batch_size: int = 256,
    seed: int = 42,
) -> dict:
    import sys

    sys.path.insert(0, REMOTE_ROOT)

    import scripts.flow_train as flow_train

    flow_train.SEED = seed
    flow_train.BATCH_SIZE = batch_size
    flow_train.EPOCHS = epochs
    flow_train.ROOT = Path(REMOTE_ROOT)

    print("starting remote flow training")
    flow_train.main()

    checkpoint_path = Path(REMOTE_ROOT) / "outputs" / "checkpoints" / "flow_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Flow checkpoint not found after training: {checkpoint_path}")

    print(f"finished flow training, best checkpoint saved to {checkpoint_path}")
    return {"checkpoint_bytes": checkpoint_path.read_bytes()}


@app.local_entrypoint()
def main(
    epochs: int = 6,
    batch_size: int = 256,
    seed: int = 42,
) -> None:
    result = run_backbone_training.remote(
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )

    output_dir = LOCAL_ROOT / "outputs" / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "flow_best.pt"
    checkpoint_path.write_bytes(result["checkpoint_bytes"])

    print("remote flow training complete")
    print(f"checkpoint_saved={checkpoint_path}")
