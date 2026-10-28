import numpy as np

from dataclasses import dataclass, field
from typing import Any, Optional

from src.simulation.runner import (
    simulate_continuous_task,
    simulate_independent_trials, extract_continuous_features
)

from src.simulation.evaluation import (
    evaluate_continuous_regression,
    evaluate_scalar_classification,
    evaluate_independent_classification,
)

from src.utils.features import build_mpsk_feature_matrix
from src.tasks.properties.memory_capacity import evaluate_linear_memory_capacity


# ============================================================
# Experiment configuration
# ============================================================

@dataclass
class ExperimentConfig:

    # Simulation
    symbol_period: Optional[float] = None
    points_per_symbol: int = 16
    k: int = 8

    amplitude_scale: float = 1e-2
    encoding: str = "zoh"
    carrier_frequency: Optional[float] = None

    # Readout
    ridge: float = 1e-6
    normalize: bool = True

    # Evaluation
    evaluation_type: str = "regression"

    # Scalar classification
    valid_start: Optional[int] = None
    valid_stop: Optional[int] = None

    # Independent trials
    t_start: Optional[float] = None
    t_end: Optional[float] = None

    n_classes: Optional[int] = None

    solver: str = "exact_zoh"  # 新增


# ============================================================
# Experiment result
# ============================================================

@dataclass
class ExperimentResult:

    task_name: str
    reservoir_name: str

    evaluation_type: str

    config: ExperimentConfig

    simulation: Any
    evaluation: Any

    metadata: dict = field(default_factory=dict)


# ============================================================
# Readout experiment
# ============================================================

class ReadoutExperiment:

    def __init__(
        self,
        reservoir,
        task_data,
        config: ExperimentConfig,
        test_data=None,
        reservoir_name=None,
    ):
        self.reservoir = reservoir
        self.task_data = task_data
        self.test_data = test_data
        self.config = config

        self.reservoir_name = (
            reservoir_name
            if reservoir_name is not None
            else reservoir.__class__.__name__
        )

    # ========================================================
    # Simulation
    # ========================================================

    def simulate(self):
        """
        Run reservoir simulation without readout training.

        Returns
        -------
        SimulationResult or dict
            Continuous task:
                SimulationResult

            Independent trials:
                {
                    "train": SimulationResult,
                    "test": SimulationResult,
                }
        """

        cfg = self.config

        if cfg.evaluation_type == "classification":

            if self.test_data is None:
                raise ValueError(
                    "Independent classification requires test_data"
                )

            train_sim = simulate_independent_trials(
                reservoir=self.reservoir,
                task_data=self.task_data,
                extract_features=False,
            )

            test_sim = simulate_independent_trials(
                reservoir=self.reservoir,
                task_data=self.test_data,
                extract_features=False,
            )

            return {
                "train": train_sim,
                "test": test_sim,
            }

        # Continuous simulation

        if cfg.symbol_period is None:
            raise ValueError(
                "symbol_period is required for continuous tasks"
            )

        return simulate_continuous_task(
            reservoir=self.reservoir,
            task_data=self.task_data,
            symbol_period=cfg.symbol_period,
            points_per_symbol=cfg.points_per_symbol,
            k=cfg.k,
            amplitude_scale=cfg.amplitude_scale,
            encoding=cfg.encoding,
            carrier_frequency=cfg.carrier_frequency,
            solver=cfg.solver,
        )

    # ========================================================
    # Evaluation
    # ========================================================

    def evaluate(
        self,
        simulation,
        k=None,
        ridge=None,
    ):
        """
        Evaluate precomputed simulation results.

        Parameters
        ----------
        simulation :
            Output of self.simulate().

        k : int or None
            Temporal nodes for independent-trial feature extraction.
            If None, use config.k.

        ridge : float or None
            Override readout regularization.

        Returns
        -------
        ExperimentResult
        """

        cfg = self.config

        k_eval = cfg.k if k is None else k
        ridge_eval = cfg.ridge if ridge is None else ridge

        # ----------------------------------------------------
        # Independent classification
        # ----------------------------------------------------

        if cfg.evaluation_type == "classification":

            train_sim = simulation["train"]
            test_sim = simulation["test"]

            _, X_train = build_mpsk_feature_matrix(
                sols=train_sim.solutions,
                K=k_eval,
                t_start=cfg.t_start,
                t_end=cfg.t_end,
            )

            _, X_test = build_mpsk_feature_matrix(
                sols=test_sim.solutions,
                K=k_eval,
                t_start=cfg.t_start,
                t_end=cfg.t_end,
            )

            eval_result = evaluate_independent_classification(
                X_train=X_train,
                y_train=self.task_data.y,
                X_test=X_test,
                y_test=self.test_data.y,
                ridge=ridge_eval,
                normalize=cfg.normalize,
                n_classes=cfg.n_classes,
            )

            feature_dim = X_train.shape[1]
            simulation_type = "independent"

        # ----------------------------------------------------
        # Continuous regression / scalar classification
        # ----------------------------------------------------

        else:

            y = simulation.y

            # Reuse the original features when K is unchanged
            if k_eval == cfg.k:

                X = simulation.X

            else:

                X = extract_continuous_features(
                    simulation=simulation,
                    n_symbols=len(y),
                    symbol_period=cfg.symbol_period,
                    k=k_eval,
                )

            if cfg.evaluation_type == "regression":

                eval_result = evaluate_continuous_regression(
                    X=X,
                    y=y,
                    washout=self.task_data.washout,
                    train_ratio=self.task_data.train_ratio,
                    ridge=ridge_eval,
                    normalize=cfg.normalize,
                )

            elif cfg.evaluation_type == "scalar_classification":

                eval_result = evaluate_scalar_classification(
                    X=X,
                    y=y,
                    washout=self.task_data.washout,
                    train_ratio=self.task_data.train_ratio,
                    ridge=ridge_eval,
                    normalize=cfg.normalize,
                    valid_start=cfg.valid_start,
                    valid_stop=cfg.valid_stop,
                    n_classes=cfg.n_classes or 4,
                )

            else:
                raise ValueError(
                    f"Unsupported evaluation type: "
                    f"{cfg.evaluation_type}"
                )

            feature_dim = X.shape[1]
            simulation_type = "continuous"

        # ----------------------------------------------------
        # Experiment result
        # ----------------------------------------------------

        return ExperimentResult(
            task_name=self.task_data.name,
            reservoir_name=self.reservoir_name,
            evaluation_type=cfg.evaluation_type,
            config=cfg,
            simulation=simulation,
            evaluation=eval_result,
            metadata={
                "simulation_type": simulation_type,
                "feature_dim": feature_dim,
                "k": k_eval,
                "ridge": ridge_eval,
            },
        )

    # ========================================================
    # Full experiment
    # ========================================================

    def run(self):
        """
        Convenience method:
            simulate() -> evaluate()
        """

        simulation = self.simulate()

        return self.evaluate(simulation)

# ============================================================
# Property Experiment Configuration
# ============================================================

@dataclass
class PropertyExperimentConfig:

    symbol_period: float

    points_per_symbol: int = 16
    k: int = 8

    amplitude_scale: float = 1e-2
    encoding: str = "zoh"

    property_type: str = "linear_memory_capacity"

    max_delay: int = 20

    ridge: float = 1e-6
    normalize: bool = True
    solver: str = "ivp"


# ============================================================
# Property Experiment Result
# ============================================================

@dataclass
class PropertyExperimentResult:

    property_type: str

    reservoir_name: str

    config: PropertyExperimentConfig

    simulation: Any
    evaluation: Any


# ============================================================
# Property Experiment
# ============================================================

class PropertyExperiment:

    def __init__(
        self,
        reservoir,
        task_data,
        config: PropertyExperimentConfig,
        reservoir_name=None,
    ):

        self.reservoir = reservoir
        self.task_data = task_data
        self.config = config

        self.reservoir_name = (
            reservoir_name
            if reservoir_name is not None
            else reservoir.__class__.__name__
        )

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    def simulate(self):

        cfg = self.config

        return simulate_continuous_task(
            reservoir=self.reservoir,
            task_data=self.task_data,
            symbol_period=cfg.symbol_period,
            points_per_symbol=cfg.points_per_symbol,
            k=cfg.k,
            amplitude_scale=cfg.amplitude_scale,
            encoding=cfg.encoding,
            solver=cfg.solver,
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    def evaluate(
        self,
        simulation,
        k=None,
        ridge=None,
    ):

        cfg = self.config

        k_eval = cfg.k if k is None else k
        ridge_eval = cfg.ridge if ridge is None else ridge

        if k_eval == cfg.k:
            X = simulation.X
        else:
            X = extract_continuous_features(
                simulation=simulation,
                n_symbols=len(self.task_data.u),
                symbol_period=cfg.symbol_period,
                k=k_eval,
            )

        if cfg.property_type == "linear_memory_capacity":

            evaluation = evaluate_linear_memory_capacity(
                X=X,
                u=self.task_data.u,
                max_delay=cfg.max_delay,
                washout=self.task_data.washout,
                train_ratio=self.task_data.train_ratio,
                ridge=ridge_eval,
                normalize=cfg.normalize,
            )

        else:
            raise ValueError(
                f"Unsupported property: {cfg.property_type}"
            )

        return PropertyExperimentResult(
            property_type=cfg.property_type,
            reservoir_name=self.reservoir_name,
            config=cfg,
            simulation=simulation,
            evaluation=evaluation,
        )

    # --------------------------------------------------------
    # Full experiment
    # --------------------------------------------------------

    def run(self):

        simulation = self.simulate()

        return self.evaluate(simulation)