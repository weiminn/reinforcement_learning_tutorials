# Policy Gradient Methods

Tabular methods and Function Approximators are value-based which means the policy looks at the value of the state-actions $Q$ and choose the action with optimal estimated return. In policy gradient methods, we use the function approxmiators to not predict the $Q$ values but to estimate the probabilities of taking each action:

$$
\begin{equation}
    \pi(a|s,\theta) \in [0, 1]
\end{equation}
$$

and thus the function approximator **is** the policy that gives out the distribution of probabilities over the actions rather than the values for you to read:

$$
\begin{equation}
    \pi(s,\theta) = [p(a_1),p(a_2),\cdots,p(a_n)]
\end{equation}
$$

and your action to take, $a$, is sampled from this distribution:

$$
\begin{equation}
    a \sim \pi(a|s,\theta)
\end{equation}
$$

and thus your policy and actions change more smoothly during learning,whereas in value-based methods, you manually choose the action using greedy or $e$-greedy methods:

$$
\begin{equation}
    a = \argmax_a \hat{q}(s,a).
\end{equation}
$$

## Representing Policies using Neural Networks

Neural networks can learn the weights and propagate the extracted features of the state vector to approximate the values of the state:

$$
\begin{equation}
    \hat{y} = \phi_2(\phi_1(x\cdot w_1) \cdot w_2).
\end{equation}
$$

To train the neural network to output the vector of probability, we compress the output values to values between $0$ and $1$ such that:

$$
\begin{equation}
    \hat{y} = [0, 1]^n
\end{equation}
$$

where $n$ is the number of possible actions to take.

### Softmax Function

In order to achieve the compression, we apply softmax to normalize the values so that not only they compress to $[0,1]$, they also all sum up to 1:

$$
\begin{equation}
    \sigma(x_i) = \frac{e^{x_i}}{\sum e^{x_i}}
\end{equation}
$$

where the values of the individual outputs are raised to powers of $e$ to get a vector of probability distribution over the possible actions:

$$
\begin{equation}
    \pi(a|s,\theta) = [\sigma(a_1),\sigma(a_2),\cdots,\sigma(a_n)].
\end{equation}
$$

### Policy Evaluation

We define performance measure of the policy/neural network as $J(\theta)$, so that we can compare them such that:

$$
\begin{equation}
    J_{\pi_1}(\theta) > J_{\pi_2}(\theta) \Rightarrow \text{We consider }J_{\pi_1}(\theta) \text{ is better than }J_{\pi_2}(\theta).
\end{equation}
$$

As such the optimal policy has the optimal paramaters $\theta$ that give us the maximal performance:

$$
\begin{equation}
    \pi_*(a|s,\theta) = \argmax_\theta J(\theta).
\end{equation}
$$

We obtain the performance estimate $\hat{J}(\theta)$ from experience samples during the simulations, and we perform the Stochastic Gradient Ascent to improve the performance with regards to its parameters:

$$
\begin{equation}
    \theta_{t+1} = \theta_{t} + \alpha \triangledown J(\theta)
\end{equation}
$$

where

$$
\begin{equation}
    \triangledown J(\theta) = [\frac{\partial \hat{J}(\theta)}{\partial \theta_1}, \frac{\partial \hat{J}(\theta)}{\partial \theta_2}, \cdots, \frac{\partial \hat{J}(\theta)}{\partial \theta_n}].
\end{equation}
$$

## Policy Gradient Theorem

We define the performance of the policy as:

$$
\begin{equation}
    J(\theta) = v_\pi(S_0).
\end{equation}
$$

The gradient of the policy's performance can be derived to be the gradient of the actions' probabilities. It means that the policy performance $J$ changes wrt to the action probabilities $\pi$ which in turn changes wrt to the parameter weights $\theta$:

$$
\begin{equation}
    \triangledown J(\theta) \propto \sum_s \mu(s) \sum_a q_\pi(s,a) \triangledown \pi(a|s, \theta)
\end{equation}
$$

where $\mu(s)$ is the state distribution following policy $\pi$.

## REINFORCE

Combination of Policy Gradient and Monte Carlo methods where we wait until the end of the simulation to generate the actual rewards from the state and action pairs of the actual experience gathered rather than bootstrapped and temporal-differenced. Then, we adjust the weights of the neural networks using the actual rewards, and state and action vectors:

$$
\begin{equation}
    \theta_{t+1} = \theta_t +  \alpha \triangledown\hat{J}(\theta).
\end{equation}
$$

where $\hat{J}(\theta)$ is just the approximated gradient because the experience are only from 1 episode.

Using the Policy Graidient Theorem, the gradient of the performance is estimated by the gradient of the action probabilities:

$$
\begin{equation}
    \triangledown \hat{J} = \gamma^tG_t \cdot \frac{\triangledown\pi(A_t|S_t,\theta_t)}{\pi(A_t|S_t,\theta_t)}
\end{equation}
$$

where bigger return $G_t$ means that the action probability carries more importance to improving the policy performance, and the gradient of the action probability $\triangledown\pi(A_t|S_t,\theta_t)$ is normalized by $\pi(A_t|S_t,\theta_t)$ to scale down the gradients for highly popular actions.

Following the policy gradient derived, the parameter weights $\theta$ are then updated to improve the estimate policy performance:

$$
\begin{equation}
    \theta_{t+1} = \theta_t + \alpha \gamma^tG_t \cdot \frac{\triangledown\pi(A_t|S_t,\theta_t)}{\pi(A_t|S_t,\theta_t)}
\end{equation}
$$

and since $\triangledown \ln \pi(A_t|S_t, \theta) = \frac{\triangledown\pi(A_t|S_t,\theta_t)}{\pi(A_t|S_t,\theta_t)}$, the final update will be:

$$
\begin{equation}
    \theta_{t+1} = \theta_t + \alpha \gamma^tG_t \cdot \triangledown \ln \pi(A_t|S_t, \theta).
\end{equation}
$$

## Entropy Regularization

Since the action probabilities are directly outputted by the neural network, the mechanism for choosing the actions are abstracted away from us. In order to ecourage exploration, we incentivse the agent to keep the entropy of its policy as high as possible:

$$
\begin{equation}
    H(X) = -\sum_{x \in X} p(x) \cdot \ln p(x)
\end{equation}
$$

which measures the level of uncertainty of a random variable.

Thus, the uncertainty of the action to be selected by the policy for a state can be expressed as:

$$
\begin{equation}
    H_\pi(A_t) = -\sum_{x \in X} \pi(a|S_t) \cdot \ln \pi(a|S_t)
\end{equation}
$$

and the gradient of which is added to to SGD update as:

$$
\begin{equation}
    \theta_{t+1} = \theta_t + \alpha [\gamma^tG_t \cdot \triangledown \ln \pi(A_t|S_t, \theta) - \beta\triangledown H(\pi)].
\end{equation}
$$
