"""Multi-task MLP for Tox21 with configurable hidden layers."""

import torch.nn as nn


class Tox21MLP(nn.Module):
    """Shared hidden layers followed by one logit per toxicity endpoint."""

    def __init__(self, input_dim, n_tasks=12, hidden_dims=(1024, 512), dropout=0.3):    # 학습할 때 30% 뉴런을 랜덤하게 끈다
        super().__init__()                                                              # overfitting 방지 & ensemble 효과 (여러 모델을 평균내는 효과) & robustness 증가

        hidden_dims = tuple(hidden_dims)
        if not hidden_dims or any(h <= 0 for h in hidden_dims):
            raise ValueError("hidden_dims must contain at least one positive layer size.")

        layers = []
        in_dim = input_dim

        # Repeat the original hidden block for each requested layer size.
        for out_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, out_dim), # 특징추출
                nn.BatchNorm1d(out_dim),    # 안정화
                nn.ReLU(),                  # 비선형성
                nn.Dropout(dropout),        # 일반화
            ])
            in_dim = out_dim

        # One raw logit per endpoint; sigmoid is applied during prediction.
        layers.append(nn.Linear(in_dim, n_tasks))   # 마지막 레이어 (output layer) - 각 task의 logits 출력
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # logits (각 task에 대한 “raw score (점수)
