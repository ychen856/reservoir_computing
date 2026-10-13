from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass
class TaskData:
    name: str

    u: np.ndarray
    y: np.ndarray

    task_type: Literal["regression", "classification"]

    washout: int = 0
    train_ratio: float = 0.8

    target_offset: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.u = np.asarray(self.u)
        self.y = np.asarray(self.y)

        if self.u.ndim == 0 or self.y.ndim == 0:
            raise ValueError("u and y must have a sample dimension")

        if len(self.u) != len(self.y):
            raise ValueError(
                "TaskData requires aligned input and target arrays"
            )

        if self.task_type not in ("regression", "classification"):
            raise ValueError("Invalid task_type")

        if not 0 <= self.washout < len(self.u):
            raise ValueError("Invalid washout")

        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")