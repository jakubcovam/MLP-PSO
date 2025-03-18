# Multilayer perceptron (MLP) with Particle swarm optimization (PSO)

This project implements a hybrid artificial neural network (ANN) model that uses PSO to optimize the model weights. It features four MLPs, each with a different optimized objective function, whose results serve as inputs into a final, fifth MLP.

---
During the calibration period, the PSO technique is used to optimize the model weights. Several PSO variants can be selected, such as:

- constriction factor
- constant inertia weight
- random inertia weight
- chaotic inertia weight
- linearly and nonlinearly varying inertia weight
- adaptive inertia weight

---

The optimized objective criteria may include:

- mean squared error (MSE)
- mean absolute error (MAE)
- mean absolute percentage error (MAPE)
- Nash–Sutcliffe efficiency (NS)
- mean relative error (MRE)
- and others

---

The model is written in C++ using the Armadillo library.

---
