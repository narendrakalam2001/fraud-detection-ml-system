import sys
import os

# add project root to python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from simulation.transaction_simulator import simulate_transactions


if __name__ == "__main__":
    simulate_transactions(20)