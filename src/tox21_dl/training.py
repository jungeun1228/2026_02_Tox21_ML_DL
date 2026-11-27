"""Dataset, reproducibility and training utilities for the Tox21 MLP."""

import random
from copy import deepcopy
from numbers import Integral

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler

from .evaluation import evaluate_multitask
from .losses import compute_pos_weight, masked_bce_with_logits_loss
from .models import Tox21MLP


def set_seed(seed=42):  # 재현성: 같은 결과가 나오도록 (config 비교 시 특히 중요)
    """Seed Python, NumPy and PyTorch before model construction."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Tox21MLPDataset(Dataset):     # Dataset = “데이터를 하나씩 꺼내는 규칙을 정의하는 클래스” / pytorch 구조) Dataset:"데이터 하나는 이렇게 생겼다"
    """One molecule's features, targets and valid-label mask."""        # -> DataLoader: "이걸 batch로 묶어줘
                                                                        # -> 모델: "batch 받아서 학습"
    def __init__(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 2 or 0 in X.shape or 0 in y.shape:
            raise ValueError("X and y must be nonempty 2-D arrays.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of molecules.")
        if not np.isfinite(X).all():
            raise ValueError("Features must be finite.")
        if not np.all(np.isnan(y) | (y == 0) | (y == 1)):
            raise ValueError("Labels must be 0, 1 or NaN.")

        self.X = torch.tensor(X, dtype=torch.float32) # numpy → torch tensor
        if not torch.isfinite(self.X).all():
            raise ValueError("Features must remain finite after float32 conversion.")
        self.y = torch.tensor(np.nan_to_num(y, nan=0.0), dtype=torch.float32)   # NaN을 임시로 0으로 바꿈 (일단 loss계산에서 1차로 포함되어서 placeholder로 넣어두지만, 이후 계산 단계에서 mask로 걸러내짐)
        self.mask = torch.tensor(~np.isnan(y), dtype=torch.float32)  # label 있는 곳: 1 / 없는 곳: 0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx], "mask": self.mask[idx]}


class _NonSingletonBatchSampler(BatchSampler):  # singleton batch를 이전 batch에 합쳐서 BatchNorm이 학습 중에 통계값을 계산할 때 문제가 없도록 함 
    """Merge a singleton final training batch into the previous batch."""   # singleton batch: batch size가 1인 batch (BatchNorm은 batch size가 1이면 통계값 계산 불가)

    def __init__(self, sampler, batch_size):
        if len(sampler) < 2 or batch_size < 2:
            raise ValueError("BatchNorm training requires >= 2 samples and batch_size >= 2.")
        super().__init__(sampler, batch_size, drop_last=False)

    def __iter__(self):
        previous = None
        for batch in super().__iter__():
            if len(batch) == 1:
                # With >= 2 samples and batch_size >= 2, this is the last batch.
                previous.extend(batch)
            else:
                if previous is not None:
                    yield previous
                previous = batch
        if previous is not None:
            yield previous

    def __len__(self):
        return super().__len__() - int(len(self.sampler) % self.batch_size == 1)


def create_dataloader(X, y, *, batch_size=128, training, seed=42):
    """Create a seeded training loader or an ordered evaluation loader.

    Training shuffles every epoch without dropping any molecule. If the
    last batch contains one molecule, it is merged into the preceding batch.
    Evaluation retains all molecules in input order, including singletons.
    Workers are disabled for simplicity and Windows notebook compatibility.
    """
    if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    batch_size = int(batch_size)
    dataset = Tox21MLPDataset(X, y)
    loader_generator = torch.Generator().manual_seed(seed)

    if training:
        # Keep the shuffle RNG independent of model initialization and Dropout.
        shuffle_generator = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(dataset, generator=shuffle_generator)
        batch_sampler = _NonSingletonBatchSampler(sampler, batch_size)
        return DataLoader(dataset, batch_sampler=batch_sampler,
                          generator=loader_generator, num_workers=0)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      drop_last=False, generator=loader_generator, num_workers=0)


def train_one_epoch(model, loader, optimizer, device, pos_weight=None):
    """Train for one epoch and return mean weighted BCE per valid label.

    The caller places the model on device. Use a training loader created by
    create_dataloader to avoid singleton batches with BatchNorm. Skip batches
    without valid labels before the forward pass or optimizer update.
    """
    model.train()   # Dropout을 활성화하고 BatchNorm이 학습 중에 통계값을 계산하도록 함
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        # Count observed labels, not molecules.
        valid_count = int(batch["mask"].sum().item())
        if valid_count == 0:
            continue

        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()   # 이전 batch의 gradient 초기화
        logits = model(x)   # Forward pass
        loss = masked_bce_with_logits_loss( # 유효한 Label에 대한 평균 Loss를 반환
            logits, y, mask, pos_weight=pos_weight
        )
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite training loss.")
        loss.backward() # 	Backward pass를 통해 gradient 계산
        optimizer.step()    # 	모델 weight 업데이트

        total_loss += loss.item() * valid_count # .item() : tensor → float; 이 batch에서의 평균 loss를 valid label 개수로 곱해서 전체 loss를 계산
        total_count += valid_count

    if total_count == 0:
        raise ValueError("Training epoch has no valid labels.")
    return total_loss / total_count

# evaluate_loss(): train_one_epoch()랑 거의 동일 but backward propagation과 optimizer step을 수행하지 않음
@torch.no_grad()    # gradient 계산을 완전히 끄는 설정  (validation/test에서는 학습 안하는데도, gradient를 자동계산해버리면 메모리낭비, 속도 저하라서)
def evaluate_loss(model, loader, device, pos_weight=None):
    """Return mean weighted BCE per valid label without updating the model.

    If class weighting is used, pass the same training-derived pos_weight
    as in train_one_epoch. Evaluation may include singleton batches.
    """
    model.eval() # Dropout을 비활성화하고 BatchNorm이 학습 중에 기록한 통계값을 사용하도록 함 (validation/test 시에)
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        valid_count = int(batch["mask"].sum().item())
        if valid_count == 0:
            continue

        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)

        logits = model(x)
        loss = masked_bce_with_logits_loss(
            logits, y, mask, pos_weight=pos_weight
        )
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite evaluation loss.")

        total_loss += loss.item() * valid_count
        total_count += valid_count

    if total_count == 0:
        raise ValueError("Evaluation has no valid labels.")
    return total_loss / total_count


def fit_tox21_mlp(      # Early stopping을 적용한 Tox21 MLP 학습
    X_train,            # # 모델을 학습(train)하고, validation 성능을 보면서 자동으로 멈추고, 가장 좋은 모델을 반환하는 함수
    y_train,            # # 왜 하냐? 학습 계속하면, train set 성능만 좋아지고 val set 성능은 감소 (overfitting)
    X_val,              ## -> val 성능이 최고일때 멈추자.
    y_val,
    task_names,
    hidden_dims=(1024, 512),
    dropout=0.3,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-5,
    max_epochs=100,
    patience=15,        # # 매 epoch마다 validation 평가 -> 성능 안 좋아지면 patience 카운트 -> 일정 횟수 넘으면 자동 종료 (early stopping)
    selection_metric="roc_auc",                                                        # -> 가장 좋은 모델로 복원
    device=None,
    *,
    pos_weight_cap=10.0,
    seed=42,
    verbose=True,
):
    """Fit one configuration on a preprocessed training/validation partition.

    Use only training labels to calculate positive-class weights. Select
    the best epoch by validation ROC-AUC (default) or AP and restore its
    complete model state, including BatchNorm buffers. The validation set
    is used for model selection, not independent performance estimation.
    Do not pass the outer holdout test as validation. Final fixed-epoch
    training without validation is a separate workflow.

    Returns model, epoch history, per-task history, and best_info.
    Epoch numbers start at one. Ties retain the earliest best epoch.
    """
    if selection_metric not in ("roc_auc", "ap"):
        raise ValueError("selection_metric must be 'roc_auc' or 'ap'.")
    for name, value in (("max_epochs", max_epochs), ("patience", patience)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive.")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative.")

    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_val, y_val = np.asarray(X_val), np.asarray(y_val)
    task_names = list(task_names)
    if any(a.ndim != 2 or 0 in a.shape for a in (X_train, y_train, X_val, y_val)):
        raise ValueError("Features and labels must be nonempty 2-D arrays.")
    if X_train.shape[1] != X_val.shape[1] or y_train.shape[1] != y_val.shape[1]:
        raise ValueError("Training and validation dimensions must agree.")
    if len(task_names) != y_train.shape[1] or len(set(task_names)) != len(task_names):
        raise ValueError("task_names must contain one unique name per endpoint.")

    device = torch.device(device if device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    set_seed(seed)
    train_loader = create_dataloader(       # Create data loader for training and validation sets
        X_train, y_train, batch_size=batch_size, training=True, seed=seed
    )
    val_loader = create_dataloader(
        X_val, y_val, batch_size=batch_size, training=False, seed=seed
    )
    model = Tox21MLP(       # Create model
        input_dim=X_train.shape[1], n_tasks=y_train.shape[1],
        hidden_dims=hidden_dims, dropout=dropout,
    ).to(device)
    pos_weight = compute_pos_weight(y_train, cap=pos_weight_cap).to(device) # Calculate pos_weight (due to class imbalance)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # Create optimizer

    best_state = None   # Initialize early stopping variables
    best_score = -np.inf
    best_info = None
    patience_counter = 0
    history = []
    per_task_history = []
     # Training loop
    for epoch in range(1, max_epochs + 1):  
        train_loss = train_one_epoch(model, train_loader, optimizer, device, pos_weight) # 1. Train
        val_loss = evaluate_loss(model, val_loader, device, pos_weight) # 2. Evaluate with val set 
        val_summary = evaluate_multitask(model, val_loader, device, task_names)
        val_roc_auc = val_summary["mean_roc_auc"]   # (* 여기서 핵심: 최종적으로는 모델을 "loss"가 아니라 "ROC-AUC" 기준으로 평가)
        val_ap = val_summary["mean_ap"]
        score = val_roc_auc if selection_metric == "roc_auc" else val_ap
        if not np.isfinite(score):
            raise ValueError("No finite validation selection score; check endpoint class coverage.")

        # Strict improvement: keep the earliest checkpoint on ties.
        improved = score > best_score # Early Stopping logics
        if improved:
            best_score = score
            best_state = deepcopy(model.state_dict())
            best_info = {
                "best_epoch": epoch,
                "best_score": float(score),
                "best_val_roc_auc": val_roc_auc,
                "best_val_ap": val_ap,
                "best_val_loss": val_loss,
                "best_n_valid_tasks": val_summary["n_valid_tasks"],
                "selection_metric": selection_metric,
                "seed": seed,
            }
            patience_counter = 0  # reset patience
        else:
            patience_counter += 1 # when performance is not the best (when ROC-AUC does not get better)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_roc_auc": val_roc_auc,
            "val_mean_ap": val_ap,
            "n_valid_tasks": val_summary["n_valid_tasks"],
            "is_best_so_far": improved,
            "patience_counter": patience_counter,
        })
        tmp = val_summary["per_task_df"].copy()
        tmp["epoch"] = epoch
        per_task_history.append(tmp)

        if verbose:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_mean_roc_auc={val_roc_auc:.4f} | "
                  f"val_mean_ap={val_ap:.4f}" + (" | best" if improved else ""))
        if patience_counter >= patience:    # stopping condition
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state) # load the best model
    model.eval()
    best_info["epochs_run"] = len(history)
    best_info["stopped_early"] = len(history) < max_epochs

    history_df = pd.DataFrame(history)
    per_task_history_df = pd.concat(per_task_history, ignore_index=True)
    return model, history_df, per_task_history_df, best_info


def fit_final_tox21_mlp(
    X_train,
    y_train,
    task_names,
    *,
    epochs,
    hidden_dims=(1024, 512),
    dropout=0.3,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-5,
    pos_weight_cap=10.0,
    seed=42,
    device=None,
    verbose=True,
):
    """Fit a fresh model on development data for a fixed number of epochs.

    No validation, early stopping or holdout evaluation is performed.
    Return the last-epoch model, training history and training class weights.
    """
    if isinstance(epochs, bool) or not isinstance(epochs, Integral) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")

    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive.")

    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative.")

    X_train, y_train = np.asarray(X_train), np.asarray(y_train)

    task_names = list(task_names)

    if any(a.ndim != 2 or 0 in a.shape for a in (X_train, y_train)):
        raise ValueError("Features and labels must be nonempty 2-D arrays.")

    if len(task_names) != y_train.shape[1] or len(set(task_names)) != len(task_names):
        raise ValueError("task_names must contain one unique name per endpoint.")

    device = torch.device(device if device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))

    # Initialize the random state before model construction.
    set_seed(seed)

    train_loader = create_dataloader(
        X_train,
        y_train,
        batch_size=batch_size,
        training=True,
        seed=seed,
    )

    # Initialize a new model.
    model = Tox21MLP(
        input_dim=X_train.shape[1],
        n_tasks=y_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    # Compute class weights using all development labels.
    pos_weight = compute_pos_weight(
        y_train,
        cap=pos_weight_cap,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Train for the predetermined number of epochs.
    history = []

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            pos_weight,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
        })

        if verbose:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"train_loss={train_loss:.4f}"
            )

    model.eval()

    return (
        model,
        pd.DataFrame(history),
        pos_weight.detach().cpu(),
    )