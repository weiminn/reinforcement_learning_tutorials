# Dynamic Programming

## Optimal Substructure

The Optimal Solution to **all subproblems** produces the optimal solution to the original problem. In other words, if you find the **optimal policy for all states** in a problem, you find the optimal policy for **the whole problem**.

Thereforce, DP turns Bellman equations into Update Rules:

$$v_*(s)= \max_a \sum_{s', r} p(s', r|s, a)[r + \gamma v_*(s')] \text{ becomes}$$

$$v(s) \leftarrow \max_a \sum_{s',r}p(s',r|s,a)[r+\gamma v(s')]$$

You need to know in advance how the state changes and what rewards we get from performing each action in each state: $p(s',r|s,a)$.

## Value Iteration

Update $v$ iteratively to get better approximation of optimal value of **every action** at every iteration:

$$v(s) \leftarrow \max_a \sum_{s',r}p(s',r|s,a)[r+\gamma v(s')]$$

where $v(s')$ an old value of this current value table for the next state.

Throughout the value iterations, the policy is always constant (usually random uniform), and for every iteration, you evaluate the value for **every** action. Only at the end of all the iterations, you devise a Updated policy, by using the state values $V(s)$ and the models that gives transition probabilites $p(s',r'|s,a)$:

$$\pi(s) = \argmax_a \sum_{s',r}p(s',r|s,a)[r+\gamma v(s')]$$

## Policy Iteration

While value iteration iterates through all actions evenly, policy iteration iterates values and then update the policy (Policy Improvement) to carry out another "value iteration" using the newly set policy (Policy Evaluation):

$$v(s) \leftarrow \sum_a\pi(a|s) \sum_{s',r}p(s',r|s,a)[r+\gamma v(s')]$$

In policy iteration, you change the policy after every value iteration, so the policy is being updated (Improvement) across all the value iterations (Evaluation).

In every evaluation, you evaluate for only **one** action that is given by your current policy, and thus the $\sum_a\pi(a|s)$ expression instead of $\max_a$. It also means you already have a policy and you don't need to calculate the policy using the model $p$ that gives transition probabilites:

$$\pi(s) = \argmax_a Q(s,a)$$

But of course you are always welcomed to use the environmental model $p$'s transition probabilities to calculate the state-action values just like state value, but you won't be able to use it in model-free tasks:

$$\pi'(s) = \argmax_a \sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]$$

### Policy Improvement Theorem

If the new value of the state after choosing action with new policy, $q_\pi(s, \pi'(s))$, is greater than the original value, $v_\pi(s)$, then the new state value, $v_{\pi'}(s)$ is greater than original state value, $v_\pi(s)$:

$$q_\pi(s, \pi'(s)) \geq v_\pi(s) \Rightarrow v_{\pi'}(s) \geq v_\pi(s)$$

where $q_\pi(s,a) = \sum_{s',r}p(s',r|s,a)[r+\gamma v(s')]$.

Then, the policy is updated to the new argument (action) that gives better value for the state:

