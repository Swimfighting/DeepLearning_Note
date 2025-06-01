# Basics of Neural Network programming

正向传播 Forward pass/forward propagation step + 反向传播 backward pass/backward propagation

---

# Binary Classification

$x = \begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}$,   $x_1,x_2,x_3 = \text{Red, Green, Blue}$

$n_x$ 是 $x$ 的维度($x \in \mathbb{R}^{n_x}$) 以图片来说 有红绿蓝三个原色 $n_x$ = 图片大小 * 图片大小 * 3

$m$ 个sample

 $X$ 是输入的 $m$ 个 $x$ 的合集 ($X^T$ 不容易构建神经网络) $X \in \mathbb{R}^{n_x * m}$

![截圖 2025-05-28 下午5.13.25.png](Basics%20of%20Neural%20Network%20programming%202011693f24038045be49f73d38747346/%E6%88%AA%E5%9C%96_2025-05-28_%E4%B8%8B%E5%8D%885.13.25.png)

$y = [0,1]$

$Y = [ y^{(1)} y^{(2)} … y^{(m)}]$

$Y \in \mathbb{R}^{1*m}$

$Y_{shape} = (1,m)$

---

# Logistic regression

Given $x$, $x \in \mathbb{R}^ {n_x}$ , want $\hat{y} = P(y = 1 \mid x) , 0 \le \hat{y} \le 1$

Parameters: $w \in \mathbb{R}^{n_x}, b \in \mathbb{R}$

❌linear output: $\hat{y} = w^Tx + b$

because we want to get a probability, output $0 \le \hat{y} \le 1$

✅ $\hat{y} = \sigma(w^Tx + b)$

![截圖 2025-05-28 下午5.44.20.png](Basics%20of%20Neural%20Network%20programming%202011693f24038045be49f73d38747346/%E6%88%AA%E5%9C%96_2025-05-28_%E4%B8%8B%E5%8D%885.44.20.png)

$z = w^Tx + b$, $\sigma(z) = \frac{1}{1+e^{-z}}$

if $z$ is large, $\sigma(z) \approx 1$

if $z$ is a large negative number, $\sigma(z) \approx 0$

---

## Loss function

**define with respect to a single training example**

❌Loss (error) function: $L(\hat{y},y) = \frac{1}{2}(\hat{y}-y)^2$ but dont usually do this because it might be non-convex and have many local optimization then cant find the global optimization. $\rightarrow$ gradient descent not work well

✅$L(\hat{y},y) = -(y\log{\hat{y}} + (1-y)\log{(1-\hat{y})})$

---

## Cost function

To train the parameters $w$ and $b$

**measures an entire training set**

$J(w,b) = \frac{1}{m} \sum ^m _{i = 1} L(\hat{y}^{(i)},y^{(i)}) = -\frac{1}{m} \sum ^m _{i = 1} (y^{(i)}\log{\hat{y}^{(i)}} + (1-y^{(i)})\log{(1-\hat{y}^{(i)})})$

---

# Gradient Descent

Repeat $w:=w-\alpha \frac{dJ(w)}{dw}$ until converge (:=更新 in code we write: $w:=w-\alpha dw$) and also $b$

---

# Backward calculation

math: derivation

$\frac{d_\text{final output variable}}{d_\text{variable}}$ 

in code: “ $d_{var}$ ” represent the derivative of the final output variable($J$)

 

![截圖 2025-06-01 下午4.00.12.png](Basics%20of%20Neural%20Network%20programming%202011693f24038045be49f73d38747346/%E6%88%AA%E5%9C%96_2025-06-01_%E4%B8%8B%E5%8D%884.00.12.png)