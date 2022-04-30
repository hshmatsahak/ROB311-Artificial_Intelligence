import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS
from support import plot_n_queens_solution

def initialize_greedy_n_queens(N: int) -> list:
    """
    This function takes an integer N and produces an initial assignment that greedily (in terms of minimizing conflicts)
    assigns the row for each successive column. Note that if placing the i-th column's queen in multiple row positions j
    produces the same minimal number of conflicts, then you must break the tie RANDOMLY! This strongly affects the
    algorithm's performance!

    Example:
    Input N = 4 might produce greedy_init = np.array([0, 3, 1, 2]), which represents the following "chessboard":

     _ _ _ _
    |Q|_|_|_|
    |_|_|Q|_|
    |_|_|_|Q|
    |_|Q|_|_|

    which has one diagonal conflict between its two rightmost columns.

    You many only use numpy, which is imported as np, for this question. Access all functions needed via this name (np)
    as any additional import statements will be removed by the autograder.

    :param N: integer representing the size of the NxN chessboard
    :return: numpy array of shape (N,) containing an initial solution using greedy min-conflicts (this may contain
    conflicts). The i-th entry's value j represents the row  given as 0 <= j < N.
    """
    # Initialize greedy placement with zeros
    greedy_init = np.zeros(N, dtype=int)

    # First queen goes in a random spot
    greedy_init[0] = np.random.randint(0, N)

    # initialize row count, diagonal down count, and diagonal up count arrays to count 
    # number of queens attacking each row, and each of the two types of diagonals
    row_counts = np.zeros(N, dtype=int)
    diag_down_counts = np.zeros(2*N+1, dtype = int)
    diag_up_counts = np.zeros(2*N+1, dtype=int)

    # update row and diagonal counts given initial queen placement
    row_counts[greedy_init[0]] = 1
    diag_down_counts[-1*greedy_init[0]+N-1] = 1
    diag_up_counts[greedy_init[0]] = 1

    # loop through all queen numbers
    for i in range(1, N):
        # Identify placement along column i with least conflict, and randomly break ties
        conflict_counts = row_counts + diag_up_counts[i:N+i] + diag_down_counts[i:N+i][::-1]
        result = np.where(conflict_counts == np.amin(conflict_counts))[0]
        greedy_init[i] = result[np.random.randint(0, len(result))]

        # update counts to account for latest queen placement
        row_counts[greedy_init[i]] += 1
        diag_down_counts[i-greedy_init[i]+N-1] += 1
        diag_up_counts[i+greedy_init[i]] += 1
    
    # all N queens have been initialized 
    return greedy_init

if __name__ == '__main__':
    # You can test your code here
    assignment = initialize_greedy_n_queens(10)
    plot_n_queens_solution(assignment)
