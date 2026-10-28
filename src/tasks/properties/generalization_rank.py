from src.tasks.properties.kernel_rank import numerical_rank

def evaluate_generalization_rank(perturbed_trial_features,relative_tolerance=1e-6):
    """Each row is final state after same base history with small perturbations."""
    return numerical_rank(perturbed_trial_features,relative_tolerance)
