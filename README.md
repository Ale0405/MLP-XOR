This project implements a **simple neural network (MLP)** to learn the XOR logical function using **Python** and **NumPy**.

## 📌 Objective

The goal of the network is to learn the behavior of the XOR function:
```
| Input x1 | Input x2 | Expected Output |
|----------|----------|------------------|
|    0     |    0     |        0         |
|    0     |    1     |        1         |
|    1     |    0     |        1         |
|    1     |    1     |        0         |
```

## ⚙️ Network Architecture

- **2 input neurons**
- **1 hidden layer** with **2 neurons**
- **1 output neuron**
- Activation function: **sigmoid**
- Training via **backpropagation**

## 📂 Main Files

- `main.py`: contains the Python code for training and evaluating the model.

## ▶️ How to Run

1. Make sure you have Python 3.x installed.
2. Clone the repository:
   ```bash
   git clone https://github.com/your-username/MLP_XOR.git
   cd MLP_XOR
   ```
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Mac/Linux
   .venv\Scripts\activate.bat  # Windows
   ```
4. Install NumPy:
   ```bash
   pip install numpy
   ```
5. Run the script:
   ```bash
   python main.py
   ```

## 🧪 Example Output

```
Input: [0 0] => Predicted Output: 0.01
Input: [0 1] => Predicted Output: 0.98
Input: [1 0] => Predicted Output: 0.98
Input: [1 1] => Predicted Output: 0.02
```

## 🧠 Topics Covered

- Artificial neural networks (MLP)
- Activation functions and their derivatives
- Backpropagation algorithm
- Loss calculation (MSE)
- NumPy operations (dot product, reshape, broadcasting)

## 🏁 Final Goal

Understand **how neural networks learn** and prepare for the next steps:
- Real-world datasets
- Using `pandas`
- Machine learning libraries like `scikit-learn`

---

> Created by Alessio Amoroso 🌱 continuously learning 💪# MLP-XOR
