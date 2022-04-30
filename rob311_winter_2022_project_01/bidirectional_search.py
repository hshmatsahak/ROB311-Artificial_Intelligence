from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def bidirectional_search(problem):
    """
        Implement a bidirectional search algorithm that takes instances of SimpleSearchProblem (or its derived
        classes) and provides a valid and optimal path from the initial state to the goal state.

        :param problem: instance of SimpleSearchProblem
        :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
                 num_nodes_expanded: number of nodes expanded by your search
                 max_frontier_size: maximum frontier size during search
    """
    ####
    #   COMPLETE THIS CODE
    ####

    # These 2 variables are self-explanatory, they are not being graded but will be useful for algorithm comparison and/or debugging
    num_nodes_expanded = 0
    max_frontier_size = 0

    # Declare/Initialize variables for forward search
    src_node = Node(None, problem.init_state, None, 0) # source node is initial node
    src_queue = deque([src_node]) # add source node to forward frontier
    src_visited = set()  # initialize forward search visited set to empty
    src_frontier_set = set([src_node.state]) # keep track of nodes in forward frontier

    # Declare/Initialize variables for backwards search
    dest_node = Node(None, problem.goal_states[0], None, 0) # destination node is initial node
    dest_queue = deque([dest_node]) # add destination node to backwards frontier
    dest_visited = set() # initialize backwards search visited set to empty
    dest_frontier_set = set([dest_node.state]) # keep track of nodes in backward frontier

    # function for forward BFS step
    def forward_bfs():
        nonlocal num_nodes_expanded, max_frontier_size # nonlocal because we want them to be updated in this inner function, but still be global variables
        curr_node = src_queue.popleft() # dequeue leftmost node in queue, standard bfs
        num_nodes_expanded += 1 # expanded curr_node, so increment by 1
        src_visited.add(curr_node.state) # count curr_node's state as visited by forward search
        src_frontier_set.remove(curr_node.state) # the expanded node is no longer part of the frontier
        for action in problem.get_actions(curr_node.state): # consider all possible actions from curr_node
            next_node = problem.get_child_node(curr_node, action) # determine next_node
            if next_node.state not in src_visited: # If we haven't already visited node obtained, add it to queue and frontier set
                src_queue.append(next_node)
                src_frontier_set.add(next_node.state)
        max_frontier_size = max(max_frontier_size, len(src_queue)+len(dest_queue)) # update max forntier size since we deleted a node and replaced by 0 or more children
    
    def backward_bfs():
        nonlocal num_nodes_expanded, max_frontier_size # nonlocal because we want them to be updated in this inner function, but still be global variables
        curr_node = dest_queue.popleft() # dequeue leftmost node in queue, standard bfs
        num_nodes_expanded += 1 # expanded curr_node, so increment by 1
        dest_visited.add(curr_node.state) # count curr_node's state as visited by backward search
        dest_frontier_set.remove(curr_node.state) # the expanded node is no longer part of the frontier
        for action in problem.get_actions(curr_node.state): # consider all possible actions from curr_node
            next_node = problem.get_child_node(curr_node, action) # determine next_node
            if next_node.state not in dest_visited: # If we haven't already visited node obtained, add it to queue and frontier set
                dest_queue.append(next_node)
                dest_frontier_set.add(next_node.state)
        max_frontier_size = max(max_frontier_size, len(src_queue)+len(dest_queue)) # update max forntier size since we deleted a node and replaced by 0 or more children

    # Actual search will be series of forward and backward steps till we intersect
    intersect = 0 # set intersect flag to 0, will change to 1 whenever we get an intersection between forward an backward frontier sets
    while src_queue and dest_queue: # while there are still nodes to expand
        forward_bfs() # perform forward step
        if src_frontier_set.intersection(dest_frontier_set) != set(): # If after forward step we find an intersection, break out of while, set flag to 1 to indicate we are done, and have not failed
            intersect = 1
            break

        backward_bfs() # perform backward step
        if src_frontier_set.intersection(dest_frontier_set) != set(): # If after backward step we find an intersection, break out of while, set flag to 1 to indicate we are done, and have not failed
            intersect = 1
            break

        # This was the missing part in the AIMA textbook. Essentially, there could be a a connection between 2 nodes, one in forward frontier and other in destination frontier
        # To ensure optimal solution, we want to identify these gaps when they exist
        for node1 in src_queue:
            for node2 in dest_queue:
                if problem.is_neighbour(node1.state, node2.state):
                    return problem.trace_path(node1, src_node.state) + problem.trace_path(node2, dest_node.state)[::-1], 0, 0 # return path by combining individual paths
    
    # Two options. 1. intersect = 1 means found a path. 2. intersect = 0 means no path exists
    if intersect == 1:
        intersect_state = list(src_frontier_set.intersection(dest_frontier_set))[0] # obtian interesection node
        src_path, dest_path = [], [] # initialize separate list of nodes

        # Loop through nodes in frontier to find the intersection node, and then trace path to source
        for node in src_queue: 
            if node.state == intersect_state: # node of intersection
                src_path = problem.trace_path(node, src_node.state) # path from source to intersection
                break

        # Loop through nodes in frontier to find the intersection node, and then trace path from destination. Same procedure as above
        for node in dest_queue: 
            if node.state == intersect_state: # node of intersection
                if node.parent: # exclude intersection node since we don't want to count it twice
                    dest_path = problem.trace_path(node.parent, dest_node.state) # obtain path from dest node to node just before intersection
                break

        path = src_path + dest_path[::-1] # combine forward and backward paths
        return path, num_nodes_expanded, max_frontier_size # return relevant variables

    # No path exists
    return [], num_nodes_expanded, max_frontier_size 

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
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    # Be sure to compare with breadth_first_search!