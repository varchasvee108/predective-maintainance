from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SENSOR_COLUMNS = [f"sensor_{idx}" for idx in range(1, 22)]
SENSOR_START_COL = 5


@dataclass(frozen=True)
class RawSplit:
    engine_id: np.ndarray
    cycle: np.ndarray
    sensors: np.ndarray
    rul: np.ndarray


@dataclass(frozen=True)
class WindowedSplit:
    x: torch.Tensor
    y: torch.Tensor
    engine_id: np.ndarray
    cycle: np.ndarray


@dataclass(frozen=True)
class SmokeData:
    train: WindowedSplit
    val: WindowedSplit
    test: WindowedSplit
    selected_sensors: list[str]
    mean: np.ndarray
    std: np.ndarray


class RULWindowDataset(Dataset):
    def __init__(self, split: WindowedSplit) -> None:
        self.x = split.x
        self.y = split.y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def load_smoke_data(
    data_dir: str | Path,
    smoke_fraction: float = 0.2,
    val_fraction: float = 0.2,
    seed: int = 42,
    window_size: int = 40,
    stride: int = 1,
    rul_cap: float = 125.0,
) -> SmokeData:
    data_dir = Path(data_dir)
    train_path = data_dir / "train_FD001.txt"
    test_path = data_dir / "test_FD001.txt"
    rul_path = data_dir / "RUL_FD001.txt"
    _require_files(train_path, test_path, rul_path)

    train_engine_id, train_cycle, train_sensors = _read_fd001(train_path)
    test_engine_id, test_cycle, test_sensors = _read_fd001(test_path)
    test_rul_offsets = _read_test_rul(rul_path)

    rng = np.random.default_rng(seed)
    engine_ids = np.array(sorted(np.unique(train_engine_id)))
    rng.shuffle(engine_ids)

    smoke_count = max(2, int(np.ceil(len(engine_ids) * smoke_fraction)))
    smoke_engines = engine_ids[:smoke_count]
    val_count = max(1, int(np.ceil(len(smoke_engines) * val_fraction)))
    val_engines = smoke_engines[:val_count]
    train_engines = smoke_engines[val_count:]
    if len(train_engines) == 0:
        train_engines = smoke_engines[1:]
        val_engines = smoke_engines[:1]

    train_raw = _subset_train(train_engine_id, train_cycle, train_sensors, train_engines, rul_cap)
    val_raw = _subset_train(train_engine_id, train_cycle, train_sensors, val_engines, rul_cap)
    test_raw = _make_test_split(test_engine_id, test_cycle, test_sensors, test_rul_offsets, rul_cap)

    selected_idx = _select_sensor_indices(train_raw.sensors)
    selected_sensors = [SENSOR_COLUMNS[idx] for idx in selected_idx]
    mean, std = _normalization_stats(train_raw.sensors[:, selected_idx])

    train_split = _build_split(train_raw, selected_idx, mean, std, window_size, stride)
    val_split = _build_split(val_raw, selected_idx, mean, std, window_size, stride)
    test_split = _build_split(test_raw, selected_idx, mean, std, window_size, stride)
    _validate_splits(train_split, val_split, test_split)

    return SmokeData(
        train=train_split,
        val=val_split,
        test=test_split,
        selected_sensors=selected_sensors,
        mean=mean,
        std=std,
    )


def _require_files(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing C-MAPSS FD001 file(s): "
            + ", ".join(missing)
            + ". Expected them under data/raw/."
        )


def _read_fd001(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.loadtxt(path, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 26:
        raise ValueError(f"{path} parsed with shape {raw.shape}, expected (*, 26).")
    sensors = raw[:, SENSOR_START_COL:]
    if not np.isfinite(sensors).all():
        raise ValueError(f"{path} contains NaN or infinity in raw sensor columns.")
    return raw[:, 0].astype(np.int64), raw[:, 1].astype(np.int64), sensors.astype(np.float32)


def _read_test_rul(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float32)
    if values.ndim != 1:
        values = values.reshape(-1)
    return values


def _subset_train(
    engine_id: np.ndarray,
    cycle: np.ndarray,
    sensors: np.ndarray,
    keep_engines: np.ndarray,
    rul_cap: float,
) -> RawSplit:
    mask = np.isin(engine_id, keep_engines)
    subset = RawSplit(
        engine_id=engine_id[mask],
        cycle=cycle[mask],
        sensors=sensors[mask],
        rul=np.zeros(int(mask.sum()), dtype=np.float32),
    )
    rul = np.empty_like(subset.rul)
    for current_engine in np.unique(subset.engine_id):
        engine_mask = subset.engine_id == current_engine
        max_cycle = subset.cycle[engine_mask].max()
        rul[engine_mask] = np.minimum(max_cycle - subset.cycle[engine_mask], rul_cap)
    split = RawSplit(subset.engine_id, subset.cycle, subset.sensors, rul.astype(np.float32))
    _assert_rul_valid(split)
    return split


def _make_test_split(
    engine_id: np.ndarray,
    cycle: np.ndarray,
    sensors: np.ndarray,
    test_rul_offsets: np.ndarray,
    rul_cap: float,
) -> RawSplit:
    rul = np.empty(len(engine_id), dtype=np.float32)
    for current_engine in np.unique(engine_id):
        offset_idx = int(current_engine) - 1
        if offset_idx < 0 or offset_idx >= len(test_rul_offsets):
            raise ValueError(f"Missing RUL_FD001 entry for test engine {current_engine}.")
        engine_mask = engine_id == current_engine
        last_cycle = cycle[engine_mask].max()
        failure_cycle = last_cycle + test_rul_offsets[offset_idx]
        rul[engine_mask] = np.minimum(failure_cycle - cycle[engine_mask], rul_cap)
    split = RawSplit(engine_id, cycle, sensors, rul)
    _assert_rul_valid(split)
    return split


def _assert_rul_valid(split: RawSplit) -> None:
    if not np.isfinite(split.rul).all():
        raise ValueError("RUL contains NaN or infinity.")
    if ((split.rul < 0) | (split.rul > 125)).any():
        raise ValueError("RUL target outside [0, 125].")
    for current_engine in np.unique(split.engine_id):
        order = np.argsort(split.cycle[split.engine_id == current_engine])
        rul = split.rul[split.engine_id == current_engine][order]
        if (np.diff(rul) > 1e-6).any():
            raise ValueError(f"RUL is not non-increasing for engine {current_engine}.")


def _select_sensor_indices(train_sensors: np.ndarray) -> np.ndarray:
    std = train_sensors.std(axis=0)
    selected = np.where(std >= 1e-6)[0]
    if selected.size == 0:
        raise ValueError("Sensor filtering removed all sensors.")
    return selected


def _normalization_stats(train_sensors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_sensors.mean(axis=0).astype(np.float32)
    std = train_sensors.std(axis=0).astype(np.float32)
    std = np.clip(std, a_min=1e-6, a_max=None)
    return mean, std


def _build_split(
    split: RawSplit,
    selected_idx: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    window_size: int,
    stride: int,
) -> WindowedSplit:
    windows: list[np.ndarray] = []
    labels: list[float] = []
    engine_ids: list[int] = []
    cycles: list[int] = []

    for current_engine in np.unique(split.engine_id):
        engine_mask = split.engine_id == current_engine
        order = np.argsort(split.cycle[engine_mask])
        features = split.sensors[engine_mask][:, selected_idx][order]
        if not np.isfinite(features).all():
            raise ValueError(f"Engine {current_engine} has non-finite selected features before normalization.")
        features = (features - mean) / std
        if not np.isfinite(features).all():
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        rul = split.rul[engine_mask][order]
        cycle = split.cycle[engine_mask][order]
        if len(cycle) < window_size:
            continue
        for start in range(0, len(cycle) - window_size + 1, stride):
            end = start + window_size
            windows.append(features[start:end])
            labels.append(float(rul[end - 1]))
            engine_ids.append(int(current_engine))
            cycles.append(int(cycle[end - 1]))

    if not windows:
        x = torch.empty((0, window_size, len(selected_idx)), dtype=torch.float32)
        y = torch.empty((0,), dtype=torch.float32)
    else:
        x = torch.tensor(np.stack(windows), dtype=torch.float32)
        y = torch.tensor(np.asarray(labels, dtype=np.float32), dtype=torch.float32)

    return WindowedSplit(
        x=x,
        y=y,
        engine_id=np.asarray(engine_ids, dtype=np.int64),
        cycle=np.asarray(cycles, dtype=np.int64),
    )


def _validate_splits(*splits: WindowedSplit) -> None:
    feature_counts = {int(split.x.shape[-1]) for split in splits}
    if len(feature_counts) != 1:
        raise ValueError(f"Feature count differs across splits: {feature_counts}")
    for name, split in zip(("train", "val", "test"), splits):
        if len(split.x) == 0:
            raise ValueError(f"{name} split has zero windows.")
        if not torch.isfinite(split.x).all():
            raise ValueError(f"{name} X contains NaN or infinity.")
        if not torch.isfinite(split.y).all():
            raise ValueError(f"{name} y contains NaN or infinity.")
        if ((split.y < 0) | (split.y > 125)).any():
            raise ValueError(f"{name} y contains targets outside [0, 125].")
