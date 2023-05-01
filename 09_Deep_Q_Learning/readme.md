# Deep Q Learning

Just like [Deep SARSA](https://github.com/weiminn/reinforcement/blob/main/08_Deep_SARSA/readme.md), we have a neural network to predict the $Q$ value of state-actions given a state input. But we will have 2 seperates policies: $b$ which is an $\epsilon$-greedy policy for exploration and $\pi$ which is a greedy (target) policy for exploitation. The target policy will sample from the target network when estimating the $Q$ values for bootstrapping:

$$
\begin{equation}
    \hat{L}(w) = \frac{1}{|K|} \sum_{i=1}^{|K|} [R_i + \gamma \max_a \hat{q}(S'_{i}, A'_{i}|\theta_{target}) - \hat{q}(S_{i}, A_{i}|\theta)]^2.
\end{equation}
$$

where $|K|$ is the training batch size, and $\pi(s|\theta_{target}) = \max_a \hat{q}(S'_{i}, A'_{i}|\theta_{target})$ is the action chosen by the target policy from the target network.

Afterwards, all the other steps regarding Replay Buffer sampling, Stochastic Gradient Descent, and cloning and updating target network is the same with Deep SARSA.