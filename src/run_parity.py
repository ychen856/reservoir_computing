from src.tasks.benchmarks.parity import generate_parity
from src._remaining_runner import continuous_experiment

if __name__=='__main__':
    for window in (2,3,5,10):
        task=generate_parity(window=window)
        continuous_experiment(task,f'parity_w{window}',history=max(10,window),polynomial=True)
