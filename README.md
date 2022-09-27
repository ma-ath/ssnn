# State-space Neural Networks

## Summary

A state-space neural network is a cell represented by the equation system

$$
\begin{cases}
x[n+1] & = Ax[n]+Bu[n]\\
y[n] & = Cx[n] + Du[n],\\
\end{cases} 
$$

where $u[n]$ is the system input, $y[n]$ is its output, $x[n]$ is its internal state, and $A$, $B$, $C$ and $D$ are parameter matrices. It is a special recurent neural network that resambles a state-space control system.

In this experiment, we implement and optimize this cell using a stochastic gradient descent algorithm, in order to replicate a given input sequence. The main objectives are:

- To understand if and how the optimization converges
- To understand how sensible is the cell to state initialization
- To understand how the state variable size influences optimization
- To analize the system matrices, and
- To compare it to an LSTM cell.

## [Experiment 1 - Convergence](./experiment1.ipynb)

We realize the model with 10 state variables ($x \in \real^{10}$), and train it using SGD without momentum, in minibatchs of 32. We test different inputs 
- Pure sine $y[n] = A\sin{\omega n}$
- 3 sine harmonics $y[n] = A_1\sin{\omega_1 n} + A_2\sin{\omega_2 n} + A_3\sin{\omega_3 n}$


## Findings

### Experiment 1

The network does encounter convergence difficults, even for relativitly simple signals. 

## Discussion