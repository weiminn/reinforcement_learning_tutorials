# Dynamic Programming

## Optimal Substructure

The Optimal Solution to **all subproblems** produces the optimal solution to the original problem. In other words, if you find the **optimal policy for all states** in a problem, you find the optimal policy for **the whole problem**.

Thereforce, DP turns Bellman equations into Update Rules:

$v_*(s)= \max_a \sum_{s', r} p(s', r|s, a)[r + \gamma v_*(s')]$ becomes

$v(s) \leftarrow \max_a \sum_{s',r}p(s',r|s,a)[r+\gamma v(s')]$

You need to know in advance how the state changes and what rewards we get from performing each action in each state: $p(s',r|s,a)$.

## Value Iteration

Update $v$ iteratively to get better approximation of optimal value at every iteration:

$v(s) \leftarrow \max_a \sum_{s',r}p(s',r|s,a)[r+\gamma v(s')]$

where $v(s')$ an old value of this current value table for the next state.

## Policy Iteration

While value iteration iterates through all actions evenly, policy iteration iterates values and then set the policy to carry on iterating values using the newly set policy (Policy Evaluation):

$v(s) \leftarrow \sum_a\pi(a|s) \sum_{s',r}p(s',r|s,a)[r+\gamma v(s')]$

Getting the value of a state is now contingent on the probabiilty distribution from the policy, $\pi$, at the end of every value iteration instead of just simply getting the max at the end of value iterations.