# Engineering Log & Memory

## Current Phase: [x] 1. Environment | [x] 2. Brain | [x] 3. Evolution | [x] 4. Integration

## Epiphanies & Learnings
*   Neuroevolution / Genetic Algorithm with a population of 50 networks solves CartPole-v1 extremely fast (< 1 second) without needing backpropagation or reward discounting!
*   Elitism (preserving the top 10% unmutated) guarantees monotonic fitness improvement, preventing regression across generations.
*   The 4-input vector (cart position, cart velocity, pole angle, pole angular velocity) mapped through a 16-node ReLU hidden layer and 2-node output generates sharp, responsive left/right control actions.

## Current Status
*   All success criteria met:
    - Random network population (50 agents) evaluated in CartPole-v1.
    - Genetic selection and Gaussian mutation operational.
    - Pole stays upright for the maximum 500 frames (> 200 frame requirement).
    - Champion weights saved to `best_cartpole_model.npz`.
    - Real-Time Neural Network Decision Display implemented alongside Cart-Pole simulation in `network_visualizer.py`.

## Next Immediate Step
*   Run `python network_visualizer.py` or `python train_cartpole.py --test-only` to watch the side-by-side simulation and decision display.