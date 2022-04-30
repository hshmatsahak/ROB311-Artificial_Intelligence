import queue
from random import sample
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem
# import matplotlib.pyplot as plt

def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    ####
    #   COMPLETE THIS CODE
    ####

    # The A* search in our case is similar to Djikstra, but we use the Manhattan Heuristic function.

    # Declare/Initialize A* search variables
    explored = set() # standard djikstra/A* has an explored set to track which states have been expanded
    cost_so_far = dict() # to keep minimum cost to states
    start_node = Node(None, problem.init_state, None, 0) # pretty obvious, its the start node, which has no parent or action, and has cost 0
    num_nodes_generated = 0 # number of nodes generated
    max_frontier_size = 0 # max frontier size

    frontier = queue.PriorityQueue() # initialize frontier
    frontier.put((problem.heuristic(start_node.state), start_node)) # Put start node in frontier, with priority determined by heuristic function
    
    while not frontier.empty(): # While we can still explore
        curr = frontier.get()[1] # extract node with highest priority
        explored.add(curr.state) # mark that node as explored, so we don't visit it again

        # if current node passes goal test, we are done
        if problem.goal_test(curr.state): 
            return problem.trace_path(curr, start_node.state), num_nodes_generated, max_frontier_size # return path as usual
        
        # Otherwise, consider all possible actions
        for action in problem.get_actions(curr.state):
            child_node = problem.get_child_node(curr, action) # obtain child node from current + action
            num_nodes_generated += 1 # increment num_nodes generated in accordance to rules given by handout

            # If haven't explored child yet and cost to child is minimum seen so far
            if child_node.state not in explored and (child_node.state not in cost_so_far or child_node.path_cost < cost_so_far[child_node.state]):
                cost_so_far[child_node.state] = child_node.path_cost # update cost so far
                priority = child_node.path_cost + problem.heuristic(child_node.state) # update priority
                frontier.put((priority, child_node))  # add to frontier
            max_frontier_size = max(max_frontier_size, frontier.qsize())
    # failed to find path, return None
    return None, num_nodes_generated, max_frontier_size

# Code for experiments
def experiments():
    N_vals = [300]
    n_runs = 100
    p_occs = np.linspace(0.1, 0.9, 17)
    for N in N_vals:
        print(f'N = {N}')
        results = []
        for p_occ in p_occs:
            successes = 0
            for i in range(n_runs):
                sample_problem = get_random_grid_problem(p_occ, N, N)
                successes += a_star_search(sample_problem)[1]
            results.append(successes/n_runs)
        # plt.plot(p_occs, results)
        # plt.show()


def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should be computed on your own machine. Simply
    fill in the values!

    :return: tuple containing (transition_start_probability, transition_end_probability, peak_probability)
    """
    ####
    #   REPLACE THESE VALUES
    ####
    transition_start_probability = 0.3
    transition_end_probability = 0.5
    peak_nodes_expanded_probability = 0.4
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability

if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.9
    M = 10
    N = 10
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    problem.plot_solution(path)

    # Experiment and compare with BFS
    # experiments()