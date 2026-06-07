# CUDA C++ 学习讲义

## 第一阶段：CUDA 基础与主机/设备内存管理

在传统的 CPU 编程中，程序是单线程或少量线程顺序执行的。而在 GPU 编程中，我们通过成千上万个轻量级线程同时执行相同的代码（即 **SPMD，单程序多数据** 模式）。

### 1. 核心概念
*   **Host（主机）**：指 CPU 及其内存。
*   **Device（设备）**：指 GPU 及其显存。
*   **Kernel（核函数）**：在 GPU 上执行的函数。使用 `__global__` 关键字声明。核函数必须返回 `void`。
*   **内存隔离**：CPU 无法直接访问 GPU 显存，GPU 也无法直接访问 CPU 内存。因此，CUDA 编程的标准流程通常是：
    1. 在 CPU 上准备数据（Host 内存分配与初始化）。
    2. 在 GPU 上分配显存（使用 `cudaMalloc`）。
    3. 将数据从 Host 复制到 Device（使用 `cudaMemcpy` 并指定 `cudaMemcpyHostToDevice`）。
    4. 启动 Kernel（核函数）在 GPU 上进行并行计算。
    5. 将计算结果从 Device 复制回 Host（使用 `cudaMemcpy` 并指定 `cudaMemcpyDeviceToHost`）。
    6. 释放 Host 和 Device 上分配的内存（使用 `cudaFree` 和 `free`）。

### 2. CUDA API 快速入门
*   **显存分配**：`cudaError_t cudaMalloc(void** devPtr, size_t size);` 
    `void** devPtr` 指的是传入设备指针变量的地址，`size_t size` 指的是这块显存空间的字节数。
*   **内存拷贝**：`cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);`
    *   `kind` 常用的有 `cudaMemcpyHostToDevice` 和 `cudaMemcpyDeviceToHost`。
*   **显存释放**：`cudaError_t cudaFree(void* devPtr);`
*   **核函数调用语法**：`kernelName<<<no_of_blocks, threads_per_block>>>(arguments);`

---

### 3. 第一阶段练习题
**题目要求：**
编写一个 CUDA 程序，实现对一个长度为 $N=1024$ 的单精度浮点数数组 $A$ 的每个元素进行**平方运算**。即计算 $B[i] = A[i] \times A[i]$。
*   在 CPU 上初始化数组 $A$（例如 $A[i] = i$）。
*   在 GPU 上分配对应的显存。
*   编写一个简单的核函数，利用 $1024$ 个 GPU 线程并行完成这个计算（此时我们暂且假设使用 1 个 Block，其中包含 1024 个 Thread）。
*   将结果 $B$ 传回 CPU 并打印前 10 个元素进行验证。

---

### 4. 练习题讲解与完整代码

#### 【原理解析】
1.  **核函数声明**：
    ```cpp
    __global__ void vectorSquare(float *d_out, float *d_in, int N) {
        // threadIdx.x 是 CUDA 内置变量，代表当前线程在当前线程块（Block）中的局部索引。
        int idx = threadIdx.x; 
        if (idx < N) {
            d_out[idx] = d_in[idx] * d_in[idx];
        }
    }
    ```
2.  **线程配置**：
    因为 $N = 1024$，且当前主流 GPU 的单个 Block 最大支持 1024 个线程。我们可以配置为 `<<<1, 1024>>>`，即 1 个 Block，包含 1024 个线程。

#### 【完整代码实现】
你可以将以下代码保存为 `vector_square.cu`，并使用 NVIDIA 编译器 `nvcc vector_square.cu -o vector_square` 进行编译运行。

```cpp
#include <iostream>
#include <cuda_runtime.h>

// 1. 定义核函数
__global__ void vectorSquare(float *d_out, float *d_in, int N) {
    // 获取当前线程在 Block 中的索引
    int idx = threadIdx.x;
    
    // 防止越界（虽然这里正好是1024，但加入边界判断是极佳的编程习惯）
    if (idx < N) {
        d_out[idx] = d_in[idx] * d_in[idx];
    }
}

int main() {
    const int N = 1024;
    size_t size = N * sizeof(float);

    // 2. 分配 Host（CPU）内存并初始化
    float *h_in = new float[1024];
    float *h_out = new float[1024];

    for (int i = 0; i < N; i++) {
        h_in[i] = static_cast<float>(i);
    }

    // 3. 分配 Device（GPU）显存
    float *d_in = nullptr;
    float *d_out = nullptr;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    // 4. 将数据从 Host 拷贝到 Device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // 5. 调用核函数
    // 启动 1 个 Block，每个 Block 包含 1024 个线程
    vectorSquare<<<1, 1024>>>(d_out, d_in, N);

    // 等待 GPU 计算完成（同步）
    cudaDeviceSynchronize();

    // 6. 将计算结果从 Device 拷贝回 Host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // 7. 验证结果（打印前10个元素）
    std::cout << "Verification (First 10 elements):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << h_in[i] << " ^ 2 = " << h_out[i] << std::endl;
    }

    // 8. 释放内存
    delete[] h_in;
    delete[] h_out;
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
```

---

## 第二阶段：线程网格与任意规模数据处理

在实际应用中，处理的数据量往往远大于 1024。由于硬件限制，单个线程块（Block）最大只能容纳 1024 个线程。如果我们要处理 $N = 10^6$ 甚至更大规模的数据，就必须使用多个线程块，这构成了一个**线程网格（Grid）**。

### 1. 核心概念：线程层次结构
CUDA 组织线程的结构是两级的：
*   **Grid（网格）**：由多个 Block 组成。
*   **Block（线程块）**：由多个 Thread 组成。

为了在核函数中精确定位每一个线程，CUDA 提供了以下内置变量（它们是结构体，包含 `.x`, `.y`, `.z` 成员，支持多维，这里我们先讨论 1 维情况）：
*   `threadIdx.x`：当前线程在当前 Block 中的局部索引。
*   `blockIdx.x`：当前 Block 在整个 Grid 中的索引。
*   `blockDim.x`：一个 Block 中包含的线程数量（即线程块的维度）。
*   `gridDim.x`：一个 Grid 中包含的 Block 数量。

### 2. 全局线程索引公式
要在 1 维网格中计算出当前线程在整个 GPU 任务中的**全局唯一索引（Global Index）**，公式如下：
$$\text{global\_idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$

*理解这个公式*：你可以想象成排队，每个 Block 是一个队伍，每个队伍有 `blockDim.x` 个人。你是第 `blockIdx.x` 个队伍里的第 `threadIdx.x` 个人。在你前面的队伍一共有 `blockIdx.x` 个，每个队有 `blockDim.x` 人，所以你前面排了 `blockIdx.x * blockDim.x` 个人，再加上你在自己队伍里的位置 `threadIdx.x`，就是你的全局序号。

### 3. 如何计算需要多少个 Block？
如果数据大小为 $N$，每个 Block 包含 $T$ 个线程（通常设为 128, 256 或 512）：
*   理想情况下，我们需要 $\frac{N}{T}$ 个 Block。
*   若 $N$ 不能被 $T$ 整除，为了不漏掉尾部的数据，我们需要向上取整：
    $$\text{numBlocks} = \frac{N + T - 1}{T}$$
    在 C++ 中，这种整数除法向上取整的写法为：`int numBlocks = (N + blockDim - 1) / blockDim;`

**注意**：因为向上取整后，总线程数会略大于 $N$，所以在核函数中必须进行边界检查：`if (idx < N)`。

---

### 4. 第二阶段练习题
**题目要求：**
实现一个可以处理**任意规模**数据的数组对应元素相乘（Hadamard 乘积）程序。

*   定义两个数组 $A$ 和 $B$，长度 $N = 50,000$。
*   计算 $C[i] = A[i] \times B[i]$。
*   将每个 Block 的线程数设置为 256。
*   计算出所需的 Block 数量，并正确处理边界条件。

---

### 5. 练习题讲解与完整代码

#### 【原理解析】
1.  **Block 数量计算**：
    对于 $N = 50,000$，若每个 Block 有 256 个线程：
    $$\text{numBlocks} = (50000 + 256 - 1) / 256 = 196$$
    总线程数 $196 \times 256 = 50,176 > 50,000$。最后 176 个线程不会执行计算，因为它们通不过核函数里的 `idx < N` 检查。这能安全地避免内存非法越界访问。

2.  **核函数设计**：
    ```cpp
    __global__ void vectorMultiply(float *d_c, float *d_a, float *d_b, int N) {
        // 计算全局索引
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        // 边界保护
        if (idx < N) {
            d_c[idx] = d_a[idx] * d_b[idx];
        }
    }
    ```

#### 【完整代码实现】

```cpp
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// 核函数定义
__global__ void vectorMultiply(float *d_c, float *d_a, float *d_b, int N) {
    // 1. 计算当前线程的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. 边界检查，防止多余的线程越界访问
    if (idx < N) {
        d_c[idx] = d_a[idx] * d_b[idx];
    }
}

int main() {
    // 定义一个不能被256整除的大尺寸
    const int N = 50000;
    size_t size = N * sizeof(float);

    // 分配并初始化 Host 内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i) * 0.5f;
        h_b[i] = 2.0f; // 结果 C[i] 理论上应该等于 i
    }

    // 分配 Device 显存
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // 将数据拷贝到 GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 线程配置
    int threadsPerBlock = 256;
    // 向上取整计算 Block 数量
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching Kernel with " << blocksPerGrid << " blocks and " 
              << threadsPerBlock << " threads per block." << std::endl;

    // 启动核函数
    vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, N);

    // 同步与错误检查（在实际开发中非常有用）
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 将结果复制回 Host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 验证部分计算结果
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (std::abs(h_c[i] - static_cast<float>(i)) > 1e-5) {
            success = false;
            std::cout << "Error at index " << i << ": Expected " << i 
                      << ", but got " << h_c[i] << std::endl;
            break;
        }
    }

    if (success) {
        std::cout << "All elements calculated successfully!" << std::endl;
        std::cout << "Sample results: " << std::endl;
        std::cout << "C[100] = " << h_c[100] << " (Expected 100)" << std::endl;
        std::cout << "C[49999] = " << h_c[49999] << " (Expected 49999)" << std::endl;
    }

    // 释放资源
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```

---

## 第三阶段：多维线程网格与矩阵运算

在第一部分中，我们处理的都是一维数组。但在科学计算、图像处理和深度学习中，数据往往以二维（如图像、矩阵）或三维（如视频、体数据）的形式存在。CUDA 原生支持一维、二维和三维的线程组织。

### 1. 核心概念：`dim3` 类型与多维索引
CUDA 提供了一个特殊的三维整型向量类型 `dim3`，用于指定 Grid 和 Block 的维度。未指定的维度默认为 1。

*   **声明方式**：
    ```cpp
    dim3 threadsPerBlock(16, 16); // 16x16 的二维线程块（共 256 个线程）
    dim3 numBlocks((width + 16 - 1) / 16, (height + 16 - 1) / 16); // 二维网格
    ```
*   **多维内置变量**：
    当使用二维布局时，内置变量包含 `.x` 和 `.y` 分量：
    *   `threadIdx.x` / `threadIdx.y`
    *   `blockIdx.x` / `blockIdx.y`
    *   `blockDim.x` / `blockDim.y`

### 2. 二维图像/矩阵的映射逻辑
在计算机内存中，二维矩阵通常是以**行优先（Row-Major）**的一维连续数组形式存储的。为了将 GPU 线程映射到二维矩阵的元素上，我们约定：
*   **X 维度**对应矩阵的**列（Column / width）**，在图像中对应横坐标 $x$。
*   **Y 维度**对应矩阵的**行（Row / height）**，在图像中对应纵坐标 $y$。

根据该约定，定位矩阵元素的坐标计算公式为：
$$\text{col} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
$$\text{row} = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}$$

从而，该线程对应的**一维线性数组索引（Linear Index）**为：
$$\text{idx} = \text{row} \times \text{width} + \text{col}$$

> **重要提示**：新手极易将 `row` 和 `col` 的计算与 `x`、`y` 搞反。请牢记：**Block/Grid 的 `.x` 对应横向（列数），`.y` 对应纵向（行数）**。

---

### 3. 第三阶段练习题
**题目要求：**
编写一个 CUDA 程序，实现两个二维矩阵的相加。
*   矩阵维度：$A$ 和 $B$ 分别为 $1024 \times 2048$ 的二维矩阵（即 1024 行，2048 列）。
*   计算 $C[\text{row}][\text{col}] = A[\text{row}][\text{col}] + B[\text{row}][\text{col}]$。
*   使用 $16 \times 16$ 的二维 Block 组织线程。
*   编写核函数，处理好边界条件，将结果写回并验证。

---

### 4. 练习题讲解与完整代码

#### 【原理解析】
对于 $1024 \times 2048$ 的矩阵：
*   宽度（列数 `width`）= 2048。
*   高度（行数 `height`）= 1024。
*   Block 设为 `(16, 16)`，则：
    *   X 方向（宽度方向）需要：$(2048 + 16 - 1) / 16 = 128$ 个 Block。
    *   Y 方向（高度方向）需要：$(1024 + 16 - 1) / 16 = 64$ 个 Block。
    *   总计启动 $128 \times 64 = 8192$ 个 Block。

#### 【完整代码实现】

```cpp
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// 二维矩阵相加核函数
__global__ void matrixAdd(float *C, const float *A, const float *B, int width, int height) {
    // 计算当前线程对应的矩阵列(col)和行(row)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界条件检查（必须在矩阵有效行列范围内）
    if (row < height && col < width) {
        // 计算行优先的一维线性索引
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int height = 1024; // 行数
    const int width = 2048;  // 列数
    const int N = height * width;
    size_t size = N * sizeof(float);

    // Host 内存分配与初始化
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device 显存分配
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 拷贝数据至 GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_B, size, cudaMemcpyHostToDevice); // 注意：部分设备若此处有拼写需保持一致，d_B

    // 定义二维 Block 和 Grid 维度
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "Grid Dimensions: (" << numBlocks.x << ", " << numBlocks.y << ")" << std::endl;
    std::cout << "Block Dimensions: (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")" << std::endl;

    // 启动核函数
    matrixAdd<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, width, height);
    cudaDeviceSynchronize();

    // 拷贝结果回 Host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 验证结果
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (std::abs(h_C[i] - 3.0f) > 1e-5) {
            success = false;
            std::cout << "Error at flat index " << i << ": expected 3.0, got " << h_C[i] << std::endl;
            break;
        }
    }

    if (success) {
        std::cout << "Matrix addition verified successfully! C[0][0] = " << h_C[0] << std::endl;
    }

    // 释放资源
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
```

---

## 第四阶段：共享内存与线程同步

在之前的阶段中，所有的线程都直接从 **Global Memory（全局内存/显存）** 中读取和写入数据。Global Memory 容量大，但访问延迟非常高（通常需要几百个时钟周期）。

为了缓解这一瓶颈，GPU 在每个流多处理器（SM）上都配备了 **Shared Memory（共享内存）**。

### 1. 核心概念
*   **Shared Memory（共享内存）**：
    *   这是一种位于 GPU 芯片内部的高速缓存（片上内存），访问速度极快（接近寄存器速度）。
    *   它是**由同一个 Block 内的所有线程共享**的。不同 Block 之间的线程无法互相访问彼此的共享内存。
    *   在核函数中，使用 `__shared__` 关键字定义。
*   **线程同步 `__syncthreads()`**：
    *   当多个线程共同读写同一块共享内存时，极易产生竞争（如某些线程还没写完，另一些线程就已经开始读取）。
    *   `__syncthreads()` 是一个**栅栏（Barrier）同步点**。当一个线程执行到此处时，它必须等待**同一个 Block 内**的所有其他线程也到达此处，然后才能继续向下执行。这能确保数据的读写顺序。

---

### 2. 第四阶段练习题：一维数组滑动窗口平滑滤波（1D Stencil）
这是理解共享内存的最经典、最实用的案例。

**题目要求**：
假设有一个长度为 $N$ 的一维数组 $A$。我们希望对它进行一维平滑滤波，每个位置的输出为该位置及其左右相邻元素的平均值：
$$B[i] = \frac{A[i-1] + A[i] + A[i+1]}{3}$$
对于边界元素（$i=0$ 和 $i=N-1$），其缺失的邻居用 `0.0` 填充。

**性能挑战**：
如果采用普通方法，每个线程 $i$ 都去 Global Memory 读取 $A[i-1]$, $A[i]$, $A[i+1]$。这意味着每个数据会被重复读取 **3次**（被其自身、左邻居、右邻居各读一次）。
**优化思路**：
让每个线程只从 Global Memory 读取一次 $A[i]$ 并存入**共享内存**，然后通过 `__syncthreads()` 同步。之后，所有线程直接从高速的**共享内存**中读取邻居的数据进行计算。

由于每个 Block 需要计算相邻关系，Block 边缘的线程需要读取相邻 Block 的元素。这些处于边缘的数据称为**光晕元素（Halo Elements / Ghost Cells）**。因此，如果一个 Block 的大小是 256，其对应的共享内存大小需要声明为 `256 + 2`（左右各多出 1 个槽位）。

---

### 3. 练习题讲解与完整代码

#### 【光晕元素加载逻辑】
假设 `blockDim.x = 256`：
1.  我们声明共享内存：`__shared__ float s_data[256 + 2];`
2.  每个线程在共享内存中的对应位置（局部索引）为 `l_idx = threadIdx.x + 1`（留出 `s_data[0]` 作为左光晕）。
3.  每个线程加载自己的主数据：`s_data[l_idx] = d_in[g_idx]`。
4.  光晕加载：
    *   如果是 Block 里的第 0 个线程（`threadIdx.x == 0`），还要负责将左边的数据加载到 `s_data[0]`。
    *   如果是 Block 里的最后一个线程（`threadIdx.x == blockDim.x - 1`），还要负责将右边的数据加载到 `s_data[257]`。
5.  同步：调用 `__syncthreads()`。
6.  计算：此时安全地读取 `s_data[l_idx-1]`、`s_data[l_idx]`、`s_data[l_idx+1]` 并求和。

#### 【完整代码实现】
为了让 Block 大小在编译期确定，我们使用 C++ 模板核函数。

```cpp
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// 使用模板参数定义 Block 大小，方便在核函数中静态声明共享内存数组大小
template <int BLOCK_SIZE>
__global__ void stencil_1D(float *d_out, const float *d_in, int N) {
    // 静态分配共享内存，大小为 BLOCK_SIZE + 2（左右光晕各占1位）
    __shared__ float s_data[BLOCK_SIZE + 2];

    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int l_idx = threadIdx.x + 1; // 共享内存中留出索引0作为左边界

    // 1. 将主数据从全局内存加载到共享内存
    if (g_idx < N) {
        s_data[l_idx] = d_in[g_idx];
    } else {
        s_data[l_idx] = 0.0f; // 如果超出数组范围，填充0
    }

    // 2. 加载左、右光晕元素（Halo elements）
    // 2.1 左光晕：每个 Block 的第 0 个线程负责加载
    if (threadIdx.x == 0) {
        s_data[0] = (g_idx > 0) ? d_in[g_idx - 1] : 0.0f;
    }

    // 2.2 右光晕：每个 Block 的最后一个线程（或者数组的最后一个有效元素线程）负责加载
    if (threadIdx.x == BLOCK_SIZE - 1) {
        s_data[BLOCK_SIZE + 1] = (g_idx < N - 1) ? d_in[g_idx + 1] : 0.0f;
    }

    // 3. 栅栏同步：必须等待本 Block 内所有线程把数据装载进共享内存后，才能开始计算！
    __syncthreads();

    // 4. 从共享内存读取并执行平滑计算
    if (g_idx < N) {
        d_out[g_idx] = (s_data[l_idx - 1] + s_data[l_idx] + s_data[l_idx + 1]) / 3.0f;
    }
}

int main() {
    const int N = 1000;
    size_t size = N * sizeof(float);

    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    // 初始化输入：h_in = [0, 1, 2, 3, ...]
    for (int i = 0; i < N; i++) {
        h_in[i] = static_cast<float>(i);
    }

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // 定义配置
    const int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 启动模板核函数
    stencil_1D<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // 验证部分结果
    // 对于 i=5：(4 + 5 + 6)/3 = 5.0
    // 对于 i=0：(0 + 0 + 1)/3 = 0.333333
    // 对于 i=999 (最后一个): (998 + 999 + 0)/3 = 665.6666
    std::cout << "Verification:" << std::endl;
    std::cout << "Output[0]   = " << h_out[0] << " (Expected: " << 1.0f/3.0f << ")" << std::endl;
    std::cout << "Output[5]   = " << h_out[5] << " (Expected: 5.0)" << std::endl;
    std::cout << "Output[999] = " << h_out[999] << " (Expected: " << (998.0f + 999.0f)/3.0f << ")" << std::endl;

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);

    return 0;
}
```

欢迎进入第五阶段。在这个阶段中，我们将接触到 GPU 架构的核心秘密——**Warp（线程束）** 的执行机制，并学习如何利用它来编写并行计算中最经典的算法之一：**并行规约（Parallel Reduction）**。

---

## 第五阶段：线程束（Warp）与并行规约算法

### 1. 核心概念：什么是 Warp（线程束）？
在 GPU 硬件层面，**流多处理器（SM）** 并不是以单个 Thread 为单位来调度和执行代码的，而是以 **Warp（线程束）** 为单位。
*   **Warp 的大小**：在目前的 NVIDIA GPU 架构中，**一个 Warp 包含 32 个线程**。
*   **SIMT（单指令多线程）**：同一个 Warp 中的 32 个线程，在同一时刻只能执行**同一条相同的指令**。它们像仪仗队一样步调完全一致。

#### 线程束分化（Warp Divergence）
如果你的代码中包含分支控制语句（如 `if-else`），且同一个 Warp 中的一部分线程走 `if` 分支，另一部分走 `else` 分支，就会发生**分支分化**：
*   硬件会先让走 `if` 的线程执行，而让走 `else` 的线程处于等待（屏蔽）状态。
*   执行完 `if` 分支后，再让走 `else` 的线程执行，而屏蔽走 `if` 的线程。
*   这会导致原本应该并行的两条路径变成了**串行执行**，从而使该段代码的执行效率降低。

> **优化黄金法则**：在编写 GPU 代码时，应尽量避免让同一个 Warp 内的 32 个线程走向不同的分支。

---

### 2. 并行规约（Parallel Reduction）
**规约（Reduction）** 是一种常见的计算模式：输入一个数组（大小为 $N$），通过某种结合律算子（如加法、乘法、最大值、最小值），将其缩减为一个单一的数值。

*   **CPU 的做法**：使用一个循环，时间复杂度为 $O(N)$。
    ```cpp
    float sum = 0;
    for(int i = 0; i < N; ++i) sum += A[i];
    ```
*   **GPU 的做法（树状折叠）**：
    我们将数组数据两两相加。在第 1 步，将数组前半部分与后半部分对应相加（$N/2$ 次并行加法）；第 2 步，再折叠相加……以此类推。时间复杂度降低至 $O(\log N)$。

#### 规约中如何避免 Warp Divergence？
在进行树状折叠时，我们需要决定“哪些线程参与相加”。

*   **低效方案（交错跨步）**：
    ```cpp
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x % (2 * stride) == 0) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    ```
    *问题*：当 `stride = 1` 时，活跃的线程是 0, 2, 4, 6...。这意味着在同一个 Warp（0~31号线程）中，有的线程活跃，有的不活跃，从而导致严重的**分支分化**。

*   **高效方案（连续跨步）**：
    ```cpp
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    ```
    *优势*：当 `stride = 128` 时，活跃的线程是 0~127。这些线程在物理上是完全连续的。前 4 个 Warp（线程 0~127）完全饱满地在工作，后面的 Warp 完全空闲，从而避免了 Warp 内部的分支分化。

---

### 3. 第五阶段练习题
**题目要求：**
编写一个 CUDA 程序，计算一个大小为 $N = 1024$ 的单精度浮点数组的累加和（Sum）。
*   要求使用 **连续跨步（Strided Access）** 的并行规约算法。
*   使用 4 个 Block，每个 Block 包含 256 个线程。
*   每个 Block 负责规约自己的 256 个元素，最终输出 4 个中间结果。
*   将这 4 个中间结果拷贝回 CPU，并在 CPU 上完成最后的累加。
*   打印结果并与 CPU 串行计算的结果进行比对。

---

### 4. 练习题讲解与完整代码

#### 【算法演进图解】
假设一个 Block 负责计算其局部的 256 个元素的和，步骤如下：
1.  将当前 Block 对应的 256 个数据从全局内存载入到大小为 256 的共享内存 `sdata` 中。
2.  第一次折叠（`stride = 128`）：线程 0~127 活跃，将 `sdata[i]` 与 `sdata[i+128]` 相加，结果存入 `sdata[i]`。同步。
3.  第二次折叠（`stride = 64`）：线程 0~63 活跃，将 `sdata[i]` 与 `sdata[i+64]` 相加。同步。
4.  以此类推，直到 `stride = 1`。最终 `sdata[0]` 中保存的就是这个 Block 内 256 个元素的累加和。
5.  将 `sdata[0]` 写入全局内存的指定槽位 `d_out[blockIdx.x]`。

#### 【完整代码实现】

```cpp
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// 模板参数定义 Block 大小
template <int BLOCK_SIZE>
__global__ void blockReduceSum(float *d_out, const float *d_in, int N) {
    // 声明共享内存
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. 将数据载入共享内存
    if (gid < N) {
        sdata[tid] = d_in[gid];
    } else {
        sdata[tid] = 0.0f; // 越界部分填充0
    }

    // 确保所有线程都已将数据写入共享内存
    __syncthreads();

    // 2. 连续跨步规约（避免 Warp Divergence）
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        // 只有小于当前跨步（stride）的线程才保持活跃
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        // 每次折叠后必须同步，确保下一轮读取的都是更新后的数据
        __syncthreads();
    }

    // 3. 将本 Block 的计算结果写入全局内存
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int N = 1024;
    const int threadsPerBlock = 256;
    const int numBlocks = N / threadsPerBlock; // 4 个 Block
    size_t size_in = N * sizeof(float);
    size_t size_out = numBlocks * sizeof(float);

    // Host 内存分配
    float *h_in = (float*)malloc(size_in);
    float *h_block_sums = (float*)malloc(size_out);

    // 初始化输入数据，如所有元素初始化为 1.0
    float cpu_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f; 
        cpu_sum += h_in[i]; // 串行累加用于验证
    }

    // Device 内存分配
    float *d_in = nullptr;
    float *d_block_sums = nullptr;
    cudaMalloc((void**)&d_in, size_in);
    cudaMalloc((void**)&d_block_sums, size_out);

    // 拷贝数据至 GPU
    cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice);

    // 启动规约核函数
    blockReduceSum<threadsPerBlock><<<<numBlocks, threadsPerBlock>>>(d_block_sums, d_in, N);
    cudaDeviceSynchronize();

    // 将 4 个 Block 的中间结果拷贝回 Host
    cudaMemcpy(h_block_sums, d_block_sums, size_out, cudaMemcpyDeviceToHost);

    // 在 CPU 上对各个 Block 的中间结果做最后的累加
    float gpu_sum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        std::cout << "Block " << i << " sum = " << h_block_sums[i] << std::endl;
        gpu_sum += h_block_sums[i];
    }

    // 验证结果
    std::cout << "\n--- Verification ---" << std::endl;
    std::cout << "GPU Sum: " << gpu_sum << std::endl;
    std::cout << "CPU Sum: " << cpu_sum << std::endl;

    if (std::abs(gpu_sum - cpu_sum) < 1e-4) {
        std::cout << "SUCCESS: Results match!" << std::endl;
    } else {
        std::cout << "FAILURE: Results mismatch!" << std::endl;
    }

    // 释放资源
    free(h_in);
    free(h_block_sums);
    cudaFree(d_in);
    cudaFree(d_block_sums);

    return 0;
}
```

---

## 进阶深度思考：如何挑战极致性能？

如果你仔细观察上面的规约代码，会发现当 `stride` 降低到 32 以下时（即只有线程 0~15 活跃时）：
1.  整个 Block 只有最开头的一个 Warp 在工作。
2.  既然是在同一个 Warp 内部，所有的指令都是天然同步的（因为 SIMT 架构）。

基于这个特性，业界存在更为极致的 CUDA 优化策略：
*   **Warp Unrolling（线程束展开）**：当 `stride <= 32` 时，直接省去 `__syncthreads()` 的调用。因为单 Warp 内不需要显式屏障，这样可以极大地节省同步开销。
*   **Warp Shuffle Instructions（线程束洗牌指令）**：使用 `__shfl_down_sync()` 等硬件级指令，无需借由共享内存中转，直接在寄存器级别让 Warp 内的线程互相交换数据。

在掌握了这些核心并行思想后，你的 CUDA 基础已经非常稳固了。接下来，你可以逐步涉猎真实生产环境下的开发辅助技术，例如：
*   使用 **CUDA Stream（流）** 来实现“数据传输”与“GPU计算”的重叠并行。
*   使用 **Nsight Compute** 和 **Nsight Systems** 性能剖析工具来定位程序的性能瓶颈。

至此，我们的 CUDA 基础核心概念讲义就告一段落了。在实际开发中，保持对内存访问模式的敏感、对硬件执行单元（Warp）的理解，是编写出高质量 GPU 代码的基石。祝你在异构计算的探索中不断取得新的收获！
