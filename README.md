# CPU-vs-GPU-Matrix-Multiplication
This script will perform matrix multiplication using CPU computation with NumPy and then compare it with GPU computation using TensorFlow. 

## Matrix Multiplication Formula:
Let ![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/a656a938-915f-40ea-8705-c3f9fd15045a)
 be a ![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/e7fb6d2e-3046-4705-9628-34dfc6f6e47d)
 matrix and ![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/273910e6-c508-44b9-8770-d218d210c5ec)
 be a ![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/932c43fe-66ca-423b-bd97-913b885d0d63)
 matrix. Then the ![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/6c81b126-41b3-4170-bd96-3dcf46a65e1f)
 matrix ![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/2efdc41d-3c1f-4ca7-85be-4a4832cc0667)
 is the product of matrixes A and B and is denoted as ![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/239a6f1a-8bb4-4c01-90bd-830e9f342bf1)
, where the elements in the i th row and the j th column of the matrix ![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/8e5f2845-5650-4dc8-9750-6eac25037630)
 can be expressed as

![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/864b96b1-ba5f-4c16-b482-40fa57d9e52b)

## Calculatining the GFLOPs:

We can estimate the number of GigaFLOPS achieved for a specific matrix size and its corresponding execution time. GigaFLOPS is a common metric used to measure the performance of computations involving floating-point operations, such as matrix multiplication. Higher GigaFLOPS values generally indicate better performance, as it represents the number of operations the system can perform in one second.

**num_flops = 2 * matrix_size**3:** <br />
This calculates the total number of floating-point operations (FLOPs) required to perform matrix multiplication for two square matrices of size matrix_size x matrix_size. The matrix multiplication algorithm typically involves matrix_size**3 multiplications and matrix_size**3 - 1 additions for each element in the resulting matrix. Since we are dealing with two matrices, the total number of FLOPs is doubled, hence 2 * matrix_size**3.

**gflops = (num_flops / (execution_time * 1e9)):** <br />
This calculates the performance of the matrix multiplication operation in GigaFLOPS. execution_time represents the time taken to complete the matrix multiplication in seconds. The division num_flops / execution_time gives us the number of FLOPs per second, and dividing it by 1e9 converts it into GigaFLOPS.

## Performance Comparison:
![Screenshot](Performance_comparison.png)
