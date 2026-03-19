# Lagrangian-Neural-Network-for-Chaotic-Dynamics-double-pendulum-
This repository includes the code for synthetic data generation of a double pendulum system, and the code to model this behavior using scientific machine learning practices, specifically a Lagrangian Neural Network

📌 Project Overview: 
This repository contains the code and mathematical development for modeling the chaotic dynamics of a double pendulum using a Structured Lagrangian Neural Network (LNN).

Generic Neural Ordinary Differential Equations (Neural ODEs) often fail to conserve energy and diverge rapidly when applied to highly nonlinear, chaotic systems. This project mitigates that failure by baking physical inductive biases directly into the deep learning architecture. By learning the potential energy and a positive-definite mass matrix separately, the model successfully recovers the Euler-Lagrange equations of motion, strictly conserves total mechanical energy, and captures the geometric topology of the system's phase space.

🧮 Mathematical Development: In classical mechanics, a conservative system is defined by its Lagrangian: $$L(q, \dot{q}) = T(q, \dot{q}) - V(q)$$ Where $T$ is the kinetic energy and $V$ is the potential energy. For a double pendulum, the kinetic energy takes the specific structured form: $$T = \frac{1}{2}\dot{q}^T M(q) \dot{q}$$ Where $M(q)$ is the system's mass matrix, which is strictly symmetric and positive-definite.Instead of predicting the acceleration $\ddot{q}$ directly, this architecture utilizes two distinct multi-layer perceptrons:Potential Network: $V_\theta(q) \rightarrow \mathbb{R}$ Mass Matrix Network: $M_\phi(q) \rightarrow \mathbb{R}^{2 \times 2}$ Enforcing Positive Definiteness:To guarantee $M(q)$ remains invertible and positive-definite, the mass network outputs the non-zero elements of a lower-triangular matrix $L_{low}$ . A softplus activation with a small constant jitter ensures the diagonal elements are strictly positive. The mass matrix is then constructed via the Cholesky factorization: $$M = L_{low}L_{low}^T$$ Coordinate Embedding:To handle the periodic boundary conditions of the pendulum's angular coordinates $q$ , the inputs are embedded as $[\sin(q), \cos(q)]$ before passing through the networks.Euler-Lagrange Residual:The predicted Lagrangian $L = T - V$ is passed through PyTorch's automatic differentiation engine to compute the analytical spatial derivatives and solve for the predicted acceleration: $$M(q)\ddot{q}_{pred} = \nabla_q L - \dot{M}(q, \dot{q})\dot{q}$$ The network is optimized by minimizing the Mean Squared Error (MSE) between $\ddot{q}_{pred}$ and the finite-difference ground truth $\ddot{q}_{true}$.

📂 Repository Structure: 

generate_data.py: Solves the initial value problem for 300 random double pendulum trajectories using scipy.integrate.solve_ivp (RK45 method) with high-precision tolerances ( $10^{-9}$ ) to ensure strict energy conservation in the ground truth data. Exports to double_pendulum_dataset.csv.

train_lnn.py: Defines the Structured LNN architecture in PyTorch, computes the Euler-Lagrange residual using autograd, and trains the model utilizing a ReduceLROnPlateau learning rate scheduler.

plot_results.py: Loads the trained model weights (lnn_model.pth), integrates a test state forward in time using the learned physics engine, and generates comparative plots for time-domain divergence, phase space topology, and energy conservation.

🚀 Reproducibility (How to Run)
1. Install Dependencies
Ensure you have Python 3.8+ installed. The required packages are:

pip install torch numpy pandas scipy matplotlib


2. Generate the Ground Truth Data
Run the simulation script to create the trajectory dataset (this may take a minute or two).


python generate_data.py


3. Train the Model
Execute the training loop. The script will automatically save the best model weights to lnn_model.pth and generate a plot of the loss curve.

python train_lnn.py


4. Evaluate and Plot Results
Run the evaluation script to simulate the learned dynamics and view the phase space and energy conservation plots.

python plot_results.py
