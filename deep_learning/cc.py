import numpy as np

def cross_correlation(image, kernel, convolution=False, padding="zeroes"):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    
    padded_image = image
    
    if padding:
        padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
        padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
        if padding == "replica":
            if pad_height:
                padded_image[0:pad_height, :] = padded_image[pad_height:pad_height+1, :]
                padded_image[-pad_height:, :] = padded_image[-pad_height-1:-pad_height, :]
            if pad_width:
                padded_image[:, 0:pad_width] = padded_image[:, pad_width:pad_width+1]
                padded_image[:, -pad_width:] = padded_image[:, -pad_width-1:-pad_width]
        elif padding == "symmetry":
            if pad_height:
                padded_image[0:pad_height, :] = np.flip(padded_image[pad_height:pad_height*2, :], axis=0)
                padded_image[-pad_height:, :] = np.flip(padded_image[-pad_height*2:-pad_height, :], axis=0)
            if pad_width:
                padded_image[:, 0:pad_width] = np.flip(padded_image[:, pad_width:pad_width*2], axis=1)
                padded_image[:, -pad_width:] = np.flip(padded_image[:, -pad_width*2:-pad_width], axis=1)
        elif padding == "cyclic":
            if pad_height:
                padded_image[0:pad_height, :] = padded_image[-pad_height*2:-pad_height, :]
                padded_image[-pad_height:, :] = padded_image[pad_height:pad_height*2, :]
            if pad_width:
                padded_image[:, 0:pad_width] = padded_image[:, -pad_width*2:-pad_width]
                padded_image[:, -pad_width:] = padded_image[:, pad_width:pad_width*2]
    print(padded_image)

    if convolution:
        kernel = np.rot90(kernel, 2)

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
    return output

H = np.array([
    [10, 20, 0, 30],
    [20, 0, 10, 20],
    [10, 0, 40, 0],
    [15, 40, 0, 30]
])
print("𝐻:\n", H)
a = np.array([[-9, -1, 3, 9, 3, -1, -9]])
b = a.transpose()
x = np.dot(b, a)

# cross_correlation(H, a, padding="replica")
# cross_correlation(H, b, padding="replica")
cross_correlation(H, x, padding="cyclic")