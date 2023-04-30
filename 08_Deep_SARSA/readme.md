# Deep SARSA

The estimates of the $Q$ values are not stored in a table, but in the weights of a neural network. The state vector $s = [s_1, s_2]$ is the input to the neural network which will output $\hat{q}$ values in a vector:

$$
\begin{equation}
    \hat{q}(s) = [\hat{q}(s,a_1), \hat{q}(s,a_2), \hat{q}(s,a_3)]
\end{equation}
$$

