# part2.py: Project 4 Part 2 script
#
# --
# Artificial Intelligence
# ROB 311 Winter 2022
# Programming Project 4

import numpy as np
from mdp_env import mdp_env
from mdp_agent import mdp_agent


## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the policy_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""


def policy_iteration(env: mdp_env, agent: mdp_agent, max_iter = 1000) -> np.ndarray:
    """
    policy_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (4ed pg 657). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs-
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        max_iter: Max iterations for the algorithm

    Outputs -
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
        <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    # np.random.seed(1) # TODO: Remove this

    # Initialize variables
    # Initialize policy to random action for each state
    policy = np.random.randint(len(env.actions), size=(len(env.states), 1))

    # Initialize utility to 0 for each state
    agent.utility = np.zeros([len(env.states), 1])

    # Set counter to 0
    idx = 0

    # Set unchanged variable to True. 
    # Toggle status in while loop if best action for a state differs from current policy 
    unchanged = False
    
    # repeat till exceed max-iter or policy has converged
    while (idx < max_iter and not unchanged):
        # create copy of agent utility, which we use in for loop below
        u = np.copy(agent.utility)

        # Loop through states to update policy
        for state in env.states:
          state_probs = env.transition_model[state, :, policy[state]] # extract action probabilities
          exp_utility = np.dot(state_probs, u) # compute expected utility using action probabilities
          agent.utility[state] = env.rewards[state] + agent.gamma * exp_utility # conduct policy evaluation for current state

        # Set unchanged to True, so that after for loop below, we can check whether policy has converged or not
        unchanged = True

        # Loop through states in set of all possible states
        for state in env.states:
          # extract action probabilities as obtained from current policy
          tp_policy = env.transition_model[state, :, policy[state]]
          # compute expected utility following action probabilities above (i.e., expected utility of policy) 
          curr_val = np.dot(tp_policy.ravel(), agent.utility)

          # Now determine which action gives the best expected utility and update policy if needed

          # Extract state probabilities for each action
          tp_fromstate = env.transition_model[state, :, :]
          # Compute expected utility for each action
          options = np.matmul(agent.utility.T.ravel(), tp_fromstate)
          # Determine the maximum utility
          max_val = np.amax(options)
          # Determine is maximum utility is better than utility given by current policy
          if max_val > curr_val:
            # if so, update policy, and toggle status of unchanged variable
            policy[state] = np.where(options == max_val)[0][0]
            unchanged = False

    # return policy
    return policy.ravel()