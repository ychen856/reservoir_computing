from src.tasks.benchmarks.xor import generate_xor
from src._remaining_runner import continuous_experiment

if __name__=='__main__':
    for delay in (1,2,5,10):
        task=generate_xor(delay=delay)
        continuous_experiment(task,f'xor_d{delay}',history=max(10,delay+1),polynomial=True)
