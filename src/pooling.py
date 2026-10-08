import numpy as np 

def pooling(image, kernel_size=2, stride=2, mode="max"):

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
    output_height = (height - kernel_size) // stride + 1
    output_width  = (width - kernel_size) // stride + 1

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
            end_i = start_i + kernel_size

            end_j = start_j + kernel_size


            # # Extract pooling window
            window = image[start_i:end_i, start_j:end_j]

            if mode == "max":
                # Perform max pooling
                output[i, j] = np.max(window)
            elif mode == "avg":
                # Perform average pooling
                output[i, j] = np.mean(window)
            else:
                raise ValueError("mode must be 'max' or 'avg'")
    return output


