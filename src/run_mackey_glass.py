from src.tasks.benchmarks.mackey_glass import generate_mackey_glass
from src._remaining_runner import continuous_experiment

if __name__=='__main__':
    task=generate_mackey_glass(n_samples=3000,horizon=1)
    continuous_experiment(task,'mackey_glass_h1',history=10,polynomial=True)
