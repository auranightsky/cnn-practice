import numpy as np 

def max_pooling(image, pool_size=2, stride=2):

    """
     Max pooling formula:
     Y(i, j) = max X(i*S + m, j*S + n)
     where 0 <= m < Ph and 0 <= n < Pw

    X = input feature map
    Y= output feature map
    P_h = pooling window height
    P_w = pooling window width
    S = stride
    i,j = output coordinates
    m,n = coordinates inside the pooling window
    """
    height, width = image.shape

    # output size 
    # output size = (input size - pool size) // stride + 1
    # we use this // not this / because we want to get the whole number of output size
    output_height = (height - pool_size) // stride + 1
    output_width  = (width - pool_size) // stride + 1

    # Create an empty matrix filled with zeros, with the size of the max-pooling output.
    # we're going to put the max values into this matrix as we perform max pooling.
    output = np.zeros((output_height, output_width))

    # (i, j)
    # (0,0)  (0,1)  (0,2)
    # (1,0)  (1,1)  (1,2)
    # (2,0)  (2,1)  (2,2)
    for i in range(output_height):
        for j in range(output_width):

            # Find the coordinates of the current window
            # Based on where I am in the output, calculate where the pooling window 
            # should start in the input.
            start_i = i * stride 
            start_j = j * stride

            # where the pooling window ends
            end_i = start_i + pool_size
            end_j = start_j + pool_size

            # # Extract pooling window
            window = image[start_i:end_i, start_j:end_j]

            # Take maximun
            output[i, j] = np.max(window)
    return output


def average_pooling(image, pool_size=2, stride=2):

    pass

