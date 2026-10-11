# Project: CartPole Neural Network (From Scratch)

## Core Objective
Build a Multi-Layer Perceptron (MLP) using only NumPy to balance a simulated CartPole. Train the network using a Genetic Algorithm (evolution) instead of backpropagation.

## Tech Stack
*   **Math & Brain:** `numpy` (Strict constraint: No PyTorch, TensorFlow, or scikit-learn).
*   **Environment:** `gymnasium` (Provides the physics simulation).
*   **Rendering:** `pygame` (Used by gymnasium for visuals).

## Architecture Requirements
1.  **Input Layer:** 4 nodes (Cart Position, Cart Velocity, Pole Angle, Pole Velocity).
2.  **Hidden Layer:** 1 layer with 8-16 nodes.
3.  **Output Layer:** 2 nodes (Push Left, Push Right).
4.  **Activation Function:** ReLU for hidden, Sigmoid or Argmax for output.

## Success Criteria
*   The system can spawn a random network and run it in the environment.
*   The system can evaluate a population of 50 networks and rank them by fitness (time survived).
*   The system can mutate the best networks to create a new generation.
*   The pole stays upright for at least 200 consecutive frames (solved state).