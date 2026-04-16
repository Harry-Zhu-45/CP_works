# 计算物理第二次作业

> 祝茗 2020302191422

## 修改`EasyMatPlot.py`程序，参考matplotlib图例，绘制自己选择的函数

```python
import random
import matplotlib.pyplot as plt
import numpy as np


# 创建画布(2x2)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('test 2*2 axes')


# 子图 1
ax1.set_title('f(x)=exp(-x)*cos(2pi x)')

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)


ax1.plot(t1, f(t1), color='tab:blue', marker='o')
ax1.plot(t2, f(t2), color='black')


# 子图 2
ax2.set_title('plt.quiver')

x = np.linspace(-4, 4, 6)
y = np.linspace(-4, 4, 6)
X, Y = np.meshgrid(x, y)
U = X + Y
V = Y - X

ax2.quiver(X, Y, U, V, color="C0", angles='xy', scale_units='xy', scale=5, width=.015)
ax2.set(xlim=(-5, 5), ylim=(-5, 5))


# 子图 3
ax3.set_title('line of force')

x = np.arange(-2, 2, 0.2)
y = np.arange(-2, 2, 0.25)

X, Y = np.meshgrid(x, y)
Z = X*np.exp(-X**2-Y**2)
V, U = np.gradient(Z, 0.2, 0.2)

ax3.scatter(X, Y, color="blue", s=0.05)
ax3.quiver(X, Y, U, V, color="red", pivot="tip", units="inches")


# 子图 4
plt.axis('equal')  # axis等比例

i = 0
count = 0
n = 1000000  # 传入的总点数量

# 圆内
x_pi = []
y_pi = []

# 圆外
x_not_pi = []
y_not_pi = []

while i < n:
    x_tmp = random.random()
    y_tmp = random.random()

    if (pow(x_tmp, 2) + pow(y_tmp, 2)) < 1:
        x_pi.append(x_tmp)
        y_pi.append(y_tmp)
        count += 1
    else:
        x_not_pi.append(x_tmp)
        y_not_pi.append(y_tmp)
    i += 1

count_pi = 4 * (count / n)

ax4.scatter(x_pi, y_pi, color="blue", s=0.01)
ax4.scatter(x_not_pi, y_not_pi, color="orange", s=0.01)
ax4.set_title("Monte Carlo pi = %8.6f\nwhile n = %2.0f" % (count_pi, n))

plt.tight_layout()
plt.show()
```

> `plt.tight_layout()` 非常好用。
