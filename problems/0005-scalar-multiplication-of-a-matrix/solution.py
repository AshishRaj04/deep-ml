import numpy as np
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    # Your code here
    ele = []
    matrix = np.array(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            ele.append(matrix[i][j])
    scaled_ele = [x*scalar for x in ele]
    scaled_matrix = np.array(scaled_ele).reshape(matrix.shape)
    scaled_matrix = scaled_matrix.tolist()
    return scaled_matrix