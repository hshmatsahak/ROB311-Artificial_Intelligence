# part1_2.py: Project 4 Part 1 script
#
# --
# Artificial Intelligence
# ROB 311 Winter 2022
# Programming Project 4

from os import access
import numpy as np
from mdp_env import mdp_env
from mdp_agent import mdp_agent

## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the value_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def value_iteration(env: mdp_env, agent: mdp_agent, eps: float, max_iter = 1000) -> np.ndarray:
    """
    value_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (4ed pg 653). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs
    ---------------
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        eps:   Max error allowed in the utility of a state
        max_iter: Max iterations for the algorithm

    Outputs
    ---------------
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    # Initialize variables
    # Policy is initially empty; i.e., for each state, the best action is not known
    policy = np.empty_like(env.states)

    # Set agent utilities to 0, following AIMA Pg 653
    agent.utility = np.zeros([len(env.states), 1])

    # Set idx to 0 and increment in while loop so we do not exceed max_iter iterations
    idx = 0

    # Set delta_sat, which checks if the maximum change in utility of any state in an iteration 
    # is less than some specified amount (see formula below), to False 
    delta_sat = False

    # repeat loop until utility converges or we exceed max_iter iterations
    while (idx < max_iter and not delta_sat):
      # Set delta to 0
      delta = 0

      # Set u to be a copy of agent's utility
      u = np.copy(agent.utility).T.ravel()

      # Loop through states in set of all possible states
      for state in env.states:
        # extract 2D matrix corresponding to actions and corresponding state probabilities
        tp_fromstate = env.transition_model[state, :, :]

        # Extract summation term for each action. This corresponds to the argument of 
        # max() operator in Firgure 17.4 of AIMA Pg 653
        options = np.matmul(u, tp_fromstate)

        # Determine max value using np.amax()
        max_val = np.amax(options)

        # Determine best action by finding index of action that gave best score
        best_action = np.where(options == max_val)[0][0]

        # Update utility and policy of this state
        agent.utility[state] = env.rewards[state] + agent.gamma * max_val
        policy[state] = best_action

        # Update delta, the maximum difference in utility of any state in an iteration
        if (abs(agent.utility[state] - u[state]) > delta):
          delta = abs(agent.utility[state] - u[state])
      
      # We have completed an iteration by looping through all states.
      # Now, check if delta termination condition has been satisfied, and update if yes
      if delta < eps*(1-agent.gamma)/agent.gamma:
        delta_sat = True
      
      # Update counter
      idx+=1

    # return policy. utility not returned as we are changing it implicitly.
    return policy