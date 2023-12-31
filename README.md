# CPU-vs-GPU-Matrix-Multiplication
This script will perform matrix multiplication using CPU computation with NumPy and then compare it with GPU computation without CUDA using TensorFlow. 

## Matrix Multiplication Formula:
Let ![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/5d58481a-3ab1-4daf-ae0c-971178763d04)
 be a ![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/4c0c0c07-ecba-4c74-870a-3edf6e1ad321)
 matrix and ![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/95b1e1cd-4d36-4ae5-abc6-122dca1d0e90)
 be a ![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/a582adfd-b0c1-4427-883c-fcd8c919d71e)
 matrix. Then the ![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/7114eec2-554d-4835-9844-c34e295125b1)
 matrix ![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/fdf4e7f5-4ec5-4303-bc7d-4d3f3291c77f)
 is the product of matrixes A and B and is denoted as ![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/aa1cad57-9909-4d53-ac11-ad70578c86aa)
, where the elements in the i th row and the j th column of the matrix ![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/9a681312-1b9e-4526-ad50-05673b22dfcf)
can be expressed as

![image](https://github.com/CliveMlt/CPU-vs-GPU-Matrix-Multiplication/assets/9218133/2f956acf-828b-4de3-89fb-36276745f679)

## Calculatining the GFLOPs:

We can estimate the number of GigaFLOPS achieved for a specific matrix size and its corresponding execution time. GigaFLOPS is a common metric used to measure the performance of computations involving floating-point operations, such as matrix multiplication. Higher GigaFLOPS values generally indicate better performance, as it represents the number of operations the system can perform in one second.

**num_flops = 2 * matrix_size**3:** <br />
This calculates the total number of floating-point operations (FLOPs) required to perform matrix multiplication for two square matrices of size matrix_size x matrix_size. The matrix multiplication algorithm typically involves matrix_size**3 multiplications and matrix_size**3 - 1 additions for each element in the resulting matrix. Since we are dealing with two matrices, the total number of FLOPs is doubled, hence 2 * matrix_size**3.

**gflops = (num_flops / (execution_time * 1e9)):** <br />
This calculates the performance of the matrix multiplication operation in GigaFLOPS. execution_time represents the time taken to complete the matrix multiplication in seconds. The division num_flops / execution_time gives us the number of FLOPs per second, and dividing it by 1e9 converts it into GigaFLOPS.

## Performance Comparison:
![Screenshot](Performance_comparison.png)
