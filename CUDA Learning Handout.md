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
$$\text{globalIdx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$

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
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

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
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```
