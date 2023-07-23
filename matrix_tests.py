import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

def cpu_matrix_multiplication(matrix1, matrix2):
    return np.dot(matrix1, matrix2)

def gpu_matrix_multiplication(matrix1, matrix2):
    return tf.matmul(matrix1, matrix2)

def generate_random_matrix(size):
    return np.random.rand(size, size)

def measure_cpu_execution_time(matrix_size, num_runs=5):
    total_time = 0
    for _ in range(num_runs):
        matrix1 = generate_random_matrix(matrix_size)
        matrix2 = generate_random_matrix(matrix_size)
        
        start_time = time.time()
        cpu_matrix_multiplication(matrix1, matrix2)
        end_time = time.time()
        
        total_time += (end_time - start_time)
    
    average_time = total_time / num_runs
    return average_time

def measure_gpu_execution_time(matrix_size, num_runs=5):
    total_time = 0
    for _ in range(num_runs):
        matrix1 = generate_random_matrix(matrix_size)
        matrix2 = generate_random_matrix(matrix_size)

        matrix1_gpu = tf.constant(matrix1)
        matrix2_gpu = tf.constant(matrix2)
        
        start_time = time.time()
        gpu_matrix_multiplication(matrix1_gpu, matrix2_gpu)
        end_time = time.time()
        
        total_time += (end_time - start_time)
    
    average_time = total_time / num_runs
    return average_time

def calculate_gflops(matrix_size, execution_time):
    num_flops = 2 * matrix_size**3
    gflops = (num_flops / (execution_time * 1e9))
    return gflops

if __name__ == "__main__":
    matrix_sizes = [100, 200, 300, 400, 500, 700, 900, 1100, 1400, 1800, 2300, 3000, 4000, 5000, 6000]

    cpu_performance = []
    gpu_performance = []

    for size in matrix_sizes:
        matrix_a = generate_random_matrix(size)
        matrix_b = generate_random_matrix(size)

        average_cpu_time = measure_cpu_execution_time(size)
        average_gpu_time = measure_gpu_execution_time(size)

        gflops_cpu = calculate_gflops(size, average_cpu_time)
        gflops_gpu = calculate_gflops(size, average_gpu_time)

        cpu_performance.append(gflops_cpu)
        gpu_performance.append(gflops_gpu)

        print(f"Matrix size {size}x{size}")
        print(f"Average CPU Execution Time: {average_cpu_time:.6f} seconds")
        print(f"Average CPU Performance: {gflops_cpu:.2f} Gflops")
        print(f"Average GPU Execution Time: {average_gpu_time:.6f} seconds")
        print(f"Average GPU Performance: {gflops_gpu:.2f} Gflops\n")

    plt.figure(figsize=(8, 6))
    plt.plot(matrix_sizes, cpu_performance, label='CPU Performance', marker='o')
    plt.plot(matrix_sizes, gpu_performance, label='GPU Performance', marker='o')
    plt.xlabel('Matrix Size')
    plt.ylabel('Performance (Gflops)')
    plt.title('CPU vs. GPU Performance for Matrix Multiplication')
    plt.legend()
    plt.grid(True)
    plt.savefig('Performance_comparison.png')
    plt.show()
