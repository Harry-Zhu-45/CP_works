# 计算物理第九次作业

> 2020302191422 祝茗

## Use the symmetry of the capacitor problem (Figure 5.6) to write a program that obtains the result by calculating the potential in only one quadrant of the x-y plane

使用二维数组储存 $x-y$ 平面的数据。

```python
import numpy as np
import matplotlib.pyplot as plt


# init
s = 27
MAT = np.zeros((s, s))
```

根据对称性，边界条件与初始值可以设置为

```python
# set boundary conditions
MAT[0, :] = np.linspace(-1, 1, s)
MAT[-1, :] = np.linspace(-1, 1, s)
MAT[:, 0] = -np.ones(s)
MAT[:, -1] = np.ones(s)
```

使用 `relaxation` 方法迭代

```python
def relaxation(mat: np.ndarray, tol=1e-10, maxiter=1000) -> np.ndarray:
    """
    Relaxation method for solving Laplace's equation.
    mat: initial guess
    tol: tolerance
    maxiter: maximum number of iterations
    """
    mat_new = mat.copy()

    for i in range(maxiter):
        mat_new[1:-1, 1:-1] = 0.25 * (mat_new[:-2, 1:-1] + mat_new[2:, 1:-1] + mat_new[1:-1, :-2] + mat_new[1:-1, 2:])  # update interior points

        if np.max(np.abs(mat_new - mat)) < tol:
            break

        mat = mat_new.copy()

    return mat_new
```

计算结果，并绘制彩图

```python
MAT = relaxation(MAT)

# color map
fig, ax = plt.subplots()
im = ax.imshow(MAT, cmap='jet')
fig.colorbar(im, ax=ax)
plt.show()
```

![output](./Relaxation_method.png)

## 小黑子 pygame

门锁拍照.jpg

![jntm](./jntm.png)
