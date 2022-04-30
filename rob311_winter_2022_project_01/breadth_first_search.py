from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def breadth_first_search(problem):
    """
    Implement a simple breadth-first search algorithm that takes instances of SimpleSearchProblem (or its derived
    classes) and provides a valid and optimal path from the initial state to the goal state. Useful for testing your
    bidirectional and A* search algorithms.

    :param problem: instance of SimpleSearchProblem
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    ####
    #   COMPLETE THIS CODE
    ####

    # init_node is first node in path, it has state same as intial state, no parent or action, and path cost 0
    init_node = Node(None, problem.init_state, None, 0)

    # initialize double-ended queue, contains only initial node so far. Note: Queue = Frontier
    queue = deque([init_node])

    # Initialize visited set as empty set
    visited = set()

    # These 2 variables are self-explanatory, they are not being graded but will be useful for algorithm comparison and/or debugging
    num_nodes_expanded = 0 
    max_frontier_size = 0

    # while frontier is not empty, search
    while queue:
        max_frontier_size = max(max_frontier_size, len(queue)) # update max frontier size
        curr_node = queue.popleft() # dequeue leftmost node in queue, standard bfs algorithm
        visited.add(curr_node.state) # mark dequeued node as visited
        num_nodes_expanded += 1 # popping curr_node means you have expanded a node, so increment the counter
        if problem.goal_test(curr_node.state): # Check if we ahve found goal state
            return problem.trace_path(curr_node, init_node.state), num_nodes_expanded, max_frontier_size # If we reach here, then we have reached goal state, so simply return the traced path
        for action in problem.get_actions(curr_node.state): # If haven't reached goal state, loop through all possible actions
            next_node = problem.get_child_node(curr_node, action) # get next node from class function
            if next_node.state not in visited: # Add to frontier iff haven't visited before
                queue.append(next_node)
    return -1 # frontier empty means goal state cannot be reached, or does not exist

if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
