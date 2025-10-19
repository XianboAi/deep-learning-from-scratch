# 《深度学习入门：基于Python的理论与实现》

[TOC]

**Link**：http://www.ituring.com.cn/book/1921

<img src="./assets/image-20251018205707810.png" alt="image-20251018205707810" style="zoom:5%;" />

**表述规则**

<img src="./assets/image-20251019023611235.png" alt="image-20251019023611235" style="zoom: 33%;" />

## 第1章 Python入门

### 1.1 Python是什么

reading...

### 1.2 Python的安装

考虑create一个通用的学习环境

综合对比study、research、learn、test、main、play等名称，最终选用**lab**。



目前最新Python版本为3.13，考虑到兼容性，选择3.12版本

`conda create --name lab python=3.12`

遇到网络问题，设置USTC镜像源重新安装

<img src="./assets/image-20251018233403818.png" alt="image-20251018233403818" style="zoom:25%;" />

### 1.3 Python解释器

coding...

关闭Python解释器：Windows：Ctrl-Z  + Enter

### 1.4 Python脚本文件

考虑目录结构：

- 章节优先：/ch1/code

- **类型优先**：/code/ch1

### 1.5 Numpy

coding...

广播是通过扩展实现形状实现的

理解以下代码

```python
A = np.array([1, 2], [3, 4])
B = np.array([10, 20])
A * B
```

```python
X[X > 15]
```

### 1.6 Matplotlib

coding...

## 第2章 感知机

### 2.1 感知机是什么

reading...

### 2.2 简单逻辑电路

reading...

### 2.3 感知机的实现

coding...

`x = np.array(x1, x2)`❌

`x = np.array((x1, x2))`✔

`x = np.array([x1, x2])`✔

### 2.4 感知机的局限性

上述感知机可理解为用一条直线分割二维平面，因此无法实现异或门

### 2.5 多层感知机

单层感知机无法分离线性空间

```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
```

### 2.6 从与非门到计算机

reading...
