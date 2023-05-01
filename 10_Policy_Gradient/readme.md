# Policy Gradient Methods

Tabular methods and Function Approximators are value-based which means the policy looks at the value of the state-actions $Q$ and choose the action with optimal estimated return. In policy gradient methods, we use the function approxmiators to not predict the $Q$ values but to estimate teh probabilities of taking each action:

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

