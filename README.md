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

- To understand if the optimization converges
- To understand how sensible is the cell to state initialization
- To understand how the state variable size influences optimization
- To analize the system matrices, and
- To compare it to an LSTM cell.

## Findings

## Discussion