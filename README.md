# Ball Balancing DeepLearning 

A physics-based simulation of a **neural network–controlled agent** that balances two stacked balls on a moving cart using the **Pymunk** physics engine and **NumPy** for numerical computations. The simulation can be visualized using **Pygame** for real-time interaction and rendering.

---

## Overview

This project simulates a simple AI agent that learns to **balance two balls on a moving cart**.  
The system uses:
- A **2D physics simulation** (via Pymunk)
- A **feed-forward neural network** (from `neuralNetwork.py`)
- A **visual interface** (via Pygame)

The neural network receives continuous feedback from the environment — positions and velocities of the cart and the two balls — and outputs a control force that moves the cart to maintain balance.

## Parameters

The neural network (from neuralNetwork.py) receives six input values  
Cart position (x_cart)  
Cart velocity (x_dot_cart)  

Lower ball position (x1)  
Lower ball velocity (x1_dot)  

Upper ball position (x2)  
Upper ball velocity (x2_dot)  

It outputs a single control value force applied to the cart (F)

## Fitness Evaluation

Each agent’s fitness score increases as it stays balanced longer.
If the agent fails (balls fall or cart leaves the platform), it is destroyed and can be replaced by another (useful for evolutionary training setups).

---

## 🧠 The Agent Class

The `Agent` class encapsulates:
- A **cart** that moves horizontally on a static platform  
- Two **balls** stacked on top of each other  
- A **neural network** that decides how much force to apply to the cart  
- Logic for detecting failure (cart off screen, balls falling, etc.)  
- Visualization methods for drawing via Pygame

### Core Components
| Component | Description |
|------------|--------------|
| `__init__` | Initializes the agent, world, and shared static platform |
| `update()` | Runs one simulation step, updates neural network output |
| `disturb()` | Adds random impulses to test balancing response |
| `draw()` | Draws all shapes in the Pygame window |
| `destroy()` | Removes the agent from the physics world when dead |

---

## 🧰 Requirements

To install the dependencies, run:

```bash
pip install -r requirements.txt
```
pygame==2.6.1  
numpy>=1.26.0  
pymunk>=6.6.0  

---

## Running the Simulation

Clone this repository:
```
git clone https://github.com/yourusername/BallBalancingSimulator.git
cd BallBalancingSimulator
```

Install dependencies
```
pip install -r requirements.txt
```

Run the simulation
```
python main.py
```
The simulation window will open — watch the cart attempt to balance the two balls!

---


