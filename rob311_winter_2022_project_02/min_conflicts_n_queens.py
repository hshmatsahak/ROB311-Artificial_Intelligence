from re import S
from venv import create
import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def min_conflicts_n_queens(initialization: list):
    """
    Solve the N-queens problem with no conflicts (i.e. each row, column, and diagonal contains at most 1 queen).
    Given an initialization for the N-queens problem, which may contain conflicts, this function uses the min-conflicts
    heuristic(see AIMA, pg. 221) to produce a conflict-free solution.

    Be sure to break 'ties' (in terms of minimial conflicts produced by a placement in a row) randomly.
    You should have a hard limit of 1000 steps, as your algorithm should be able to find a solution in far fewer (this
    is assuming you implemented initialize_greedy_n_queens.py correctly).

    Return the solution and the number of steps taken as a tuple. You will only be graded on the solution, but the
    number of steps is useful for your debugging and learning. If this algorithm and your initialization algorithm are
    implemented correctly, you should only take an average of 50 steps for values of N up to 1e6.

    As usual, do not change the import statements at the top of the file. You may import your initialize_greedy_n_queens
    function for testing on your machine, but it will be removed on the autograder (our test script will import both of
    your functions).

    On failure to find a solution after 1000 steps, return the tuple ([], -1).

    :param initialization: numpy array of shape (N,) where the i-th entry is the row of the queen in the ith column (may
                           contain conflicts)

    :return: solution - numpy array of shape (N,) containing a-conflict free assignment of queens (i-th entry represents
    the row of the i-th column, indexed from 0 to N-1)
             num_steps - number of steps (i.e. reassignment of 1 queen's position) required to find the solution.
    """
    # initialize board size, num_steps and max_steps 
    N = len(initialization)
    solution = initialization.copy()
    num_steps = 0
    max_steps = 1000

    # initialize row and diagonal counts similar to greedy init    
    row_counts = np.zeros(N, dtype=int)
    diag_down_counts = np.zeros(2*N+1, dtype = int)
    diag_up_counts = np.zeros(2*N+1, dtype = int)

    # update counts given initialization
    for i in range(N):
        row_counts[initialization[i]] += 1
        diag_down_counts[i-initialization[i]+N-1] += 1
        diag_up_counts[i+initialization[i]] += 1

    # track which rows are safe
    safe = np.zeros(N, dtype=int)
    
    # update safe given initialization
    for i in range(N):
        if row_counts[initialization[i]]==1 and diag_down_counts[i-initialization[i]+N-1]==1 and diag_up_counts[i+initialization[i]]==1:
            safe[i] = 1

    # keep searching for consistent placement till you find one
    while True:
        # if all are safe, then return solution
        if np.sum(safe) == N:
            return solution, num_steps
        
        # find columns where queen in that column is not safe, and randomly sample a column from these
        conflicted = np.where(safe != 1)[0]
        col = conflicted[np.random.randint(0, len(conflicted))]

        # compute conflict count for each row in that column, identify which have least conflicts and sample one randomly from them
        conflict_counts = row_counts + diag_up_counts[col:N+col] + diag_down_counts[col:N+col][::-1]
        result = np.where(conflict_counts == np.amin(conflict_counts))[0]
        new_row = result[np.random.randint(0, len(result))]
        
        # update counts after removing queen from current row in column
        row_counts[solution[col]] -= 1
        diag_down_counts[col-solution[col]+N-1] -= 1
        diag_up_counts[col+solution[col]] -= 1

        # update counts after placing queen to new_row
        row_counts[new_row] += 1
        diag_down_counts[col-new_row+N-1] += 1
        diag_up_counts[col+new_row] += 1
        
        # update solution array to reflect replacement of queen
        solution[col] = new_row

        # re-compute boolean safe value for each queen
        for i in range(N):
            if row_counts[solution[i]]==1 and diag_down_counts[i-solution[i]+N-1]==1 and diag_up_counts[i+solution[i]]==1:
                safe[i] = 1
            else:
                safe[i] = 0

        # increment num_steps
        num_steps += 1

    return [], -1


if __name__ == '__main__':
    # Test your code here!
    from initialize_greedy_n_queens import initialize_greedy_n_queens
    from support import plot_n_queens_solution

    N = 1000
    # Use this after implementing initialize_greedy_n_queens.py
    assignment_initial = initialize_greedy_n_queens(N)
    # Plot the initial greedy assignment
    plot_n_queens_solution(assignment_initial)

    assignment_solved, n_steps = min_conflicts_n_queens(assignment_initial)
    # Plot the solution produced by your algorithm
    plot_n_queens_solution(assignment_solved)
    print(n_steps)