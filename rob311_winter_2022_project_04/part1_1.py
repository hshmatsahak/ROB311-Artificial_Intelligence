# part1_1.py: Project 4 Part 1 script
#
# --
# Artificial Intelligence
# ROB 311 Winter 2022
# Programming Project 4

import numpy as np
from mdp_cleaning_task import cleaning_env

## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the method get_transition_model which creates the
    transition probability matrix for the cleaning robot problem desribed in the
    project document.
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def get_transition_model(env: cleaning_env) -> np.ndarray:
    """
    get_transition_model method creates a table of size (SxSxA) that represents the
    probability of the agent going from s1 to s2 while taking action a
    e.g. P[s1,s2,a] = 0.5
    This is the method that will be used by the cleaning environment (described in the
    project document) for populating its transition probability table

    Inputs
    --------------
        env: The cleaning environment

    Outputs
    --------------
        P: Matrix of size (SxSxA) specifying all of the transition probabilities.
    """

    # Initialize matrix P to all-zeros matrix. Dimension of P is (num_states, num_states, num_actions). In each entry
    # P[i,j,k] we store the probability of reaching state j if we take action k from state i.
    P = np.zeros([len(env.states), len(env.states), len(env.actions)])
    
    ## START: Student Code
    
    # If 2 or less states, then just return the zero matrix, as the problem assumes we do not make a move from any of the terminal states.
    # Note that if number of states <= 2, all the states must be terminal states, justifying the condition in the if clause
    if (len(env.states)<=2):
      return P

    # Otherwise
    # Loop through all states. Note that going left takes us from state1 to state1 - 1 if successful, and going right takes us from state1 to state1 + 1 if successful.
    for state1 in env.states:
      # action = going left
      if (state1 > 0 and state1 < len(env.states)-1): # check that we are not at a terminal state
        P[state1, state1-1, 0] = 0.8 # probability of actually going left is 80%
        P[state1, state1, 0] = 0.15 # probability of staying at state1 is 15%
        P[state1, state1+1, 0] = 0.05 # probability of moving in opposite direction is 5%

      # action = going right
      if (state1 > 0 and state1 < len(env.states)-1): # check that we are not at a terminal state
        P[state1, state1+1, 1] = 0.8 # probability of actually going right is 80%
        P[state1, state1, 1] = 0.15 # probability of staying at state1 is 15%
        P[state1, state1-1, 1] = 0.05 # probability of moving in opposite direction is 5%

    ## END: Student code
    return P