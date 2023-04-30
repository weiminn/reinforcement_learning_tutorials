# Function Approximation

State Aggregation has several disadvantages:

1. Limited Precision

2. Complexity increases exponentially with dimensions of discretized states:

$$
\begin{equation}
    O(n^k)
\end{equation}
$$

where $n$ is the number of segments (aggregated discrete states) and $k$ is the number of dimensions.

Thus, we use a function $f$ that is composed of weights $w$ to approximate the value of states or state-actions. At every policy iteration, we adjust the weights until the function is sufficiently close to the optimal value function:

$$
\begin{equation}
    f_1(s|w) \rightarrow f_2(s|w) \rightarrow \cdots \rightarrow f_n(s|w) \approx v_*(s).
\end{equation}
$$

## Linear Approximators

A linear approximator consists of the sum of each dimension of the state scaled by its respective parameter:

$$
\begin{equation}
    \begin{split}
        f(s|w) &= \hat{v}(s|w) \\
               &= w \cdot s^T \\
               &= w_1 \cdot s_1 + w_2 \cdot s_2 + w_3 \cdot s_3 + \cdots + w_n \cdot s_n
    \end{split}
\end{equation}
$$

where the state is represented by vector $s$ and the parameters are represented by vector $w$ below:

$$
\begin{equation}
    s = [s_1, s_2, s_3, \cdots, s_n] \text{, and}
\end{equation}
$$

$$
\begin{equation}
    w = [w_1, w_2, w_3, \cdots, w_n].
\end{equation}
$$

## Polynomial Approximators

A polynomial approximator consists of the sum of each dimension of the state and the exponents of the component up to a certain degree scaled by its respective parameter:

$$
\begin{equation}
    \begin{split}
        f(s|w) &= \hat{v}(s|w) \\
               &= w \cdot s^T \\
               &= w_1 \cdot s_1 + w_2 \cdot s_2 + w_3 \cdot s_3 + \cdots + w_n \cdot s_n
    \end{split}
\end{equation}
$$

where the state is represented by vector $\phi(s)$ and the parameters are represented by vector $w$ below:

$$
\begin{equation}
    \phi(s) = [s_1, s^2_1, \cdots, s_1^j, s_2^1, \cdots, s_2^k, \cdots, s_m] \text{, and}
\end{equation}
$$

$$
\begin{equation}
    w = [w_1, w_2, w_3, \cdots, w_n]
\end{equation}
$$

where the each state components may have different number of exponents, and all the terms summed must be matched by the number of weights such that:

$$
\begin{equation}
    n = j + k + \cdots + m.
\end{equation}
$$

## Neural Networks

Neural networks can approximate value functions by using a set of layers that contains artificial neurons that have parameters. The linear part of the neuron aggregates the inputs into a weight sum $v_k$ w.r.t its parameters:

$$
\begin{equation}
    v_k = w_1 \cdot s_1 + w_2 \cdot s_2 + w_3 \cdot s_3 + \cdots + w_n \cdot s_n
\end{equation}
$$

which is consumed by the (usually) non-linear activation function:

$$
\begin{equation}
    y_k = \phi(v_k).
\end{equation}
$$

The whole operation can be re-expressed as:

$$
\begin{equation}
    y_k = \phi(\sum_{i=1}^nw_{k_i} \cdot x_i)
\end{equation}
$$

where $w_{k_i}$ is the $i$-th weight of the $k$-th neuron of a given layer of the neural network.

For a given input vector $x = [x_1, x_2, x_3]$, the linear/aggregation portion layer that takes in $x$ is represented by $3\times k$ matrix $w$ where $k$ is the number of neurons/outputs of this layer:

$$
\begin{equation}

[x_1, x_2, x_3]

\begin{bmatrix}
w_{11} & w_{12} & w_{13} & w_{14} & \\ 
w_{21} & w_{22} & w_{23} & w_{24} & \\ 
w_{31} & w_{32} & w_{33} & w_{34} & 
\end{bmatrix}

= [y_1, y_2, y_3, y_4]

\end{equation}
$$

where each column vector of $w$ represents the weights of the neuron of the layer.

The finally, aggregated vector goes through the activation function:

$$
\begin{equation}
    h = [\phi(y_1), \phi(y_2), \phi(y_3), \phi(y_4)].
\end{equation}
$$

Overall, the whole (2-layer for example) neural network is represented by:

$$
\begin{equation}
    \hat{y} = \phi_2(\phi_1(x\cdot w_1) \cdot w_2).
\end{equation}
$$


### Activation Functions

Activation functions helps approximate complex functions by adding non-linearity to the outputs of the neurons and, subsequently, the layers of the neural networks.

|Activation Function|Formula|Description|
|-|-|-|
|Rectifier function|$\phi(x) = \max(x,0)$|Propagates only positive outputs <br> Mainly deployed in the hidden layers to help speed up the learning process of the neural network  <br> Can approximate complex functions
|Sigmoid|$S(s) = \frac{1}{1 + e^{-x}}$|Compresses the input into $(0,1)$ <br> Usually in the last layer for classification <br> Slows down the learning if put in the hidden layers


### Stochastic Gradient Descent

Update rules for the parameters:

$$
\begin{equation}
    w_{t+1} = w_t - \alpha \triangledown \hat{L}(w)
\end{equation}
$$

where $\triangledown \hat{L}(w)$ is the gradient vector:

$$
\begin{equation}
    \triangledown \hat{L}(w) = [\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, \cdot, \frac{\partial L}{\partial w_n}]
\end{equation}
$$

where each component of the gradient vector is the partial derivative that indicates the direction to nudge each parameter of the neural network to so that the cost function $\hat{L}(w)$ increases. The parameters are updated within and across the layers using a technique called Backpropagation.


### Cost Functions

Mean Absolute error takes the average of all the errors:

$$
\begin{equation}
    L(w) = \frac{1}{M} \sum_{i=1}^N |y_i - \hat{y}_i|.
\end{equation}
$$

Mean Squared error penalized the bigger errors more by squaring the error:

$$
\begin{equation}
    L(w) = \frac{1}{M} \sum_{i=1}^N [y_i - \hat{y}_i]^2.
\end{equation}
$$

We can marry Mean Squared error with Temporal-Difference error by getting the average of all the TD-errors squared:

$$
\begin{equation}
    \hat{L}(w) = \frac{1}{M} \sum_{i=1}^N [R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}|w) - \hat{q}(S_{t}, A_{t}|w)]^2.
\end{equation}
$$

where the target is the bootstrapped $R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}|w)$, while the prediction is the current value $\hat{q}(S_{t}, A_{t}|w)$.