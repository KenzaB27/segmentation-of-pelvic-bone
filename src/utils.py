from ipywidgets import interact, fixed
from IPython.display import clear_output
import SimpleITK as sitk
import matplotlib.pyplot as plt

def read_image(file_name):
    img = sitk.ReadImage(file_name)
    img = sitk.Cast(img, sitk.sitkFloat32)
    return img


# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).


def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1, 2, figsize=(10, 8))

    # Draw the fixed image in the first subplot.
    plt.subplot(1, 2, 1)
    plt.imshow(fixed_npa[fixed_image_z, :, :], cmap=plt.cm.Greys_r)
    plt.title('fixed image')
    plt.axis('off')

    # Draw the moving image in the second subplot.
    plt.subplot(1, 2, 2)
    plt.imshow(moving_npa[moving_image_z, :, :], cmap=plt.cm.Greys_r)
    plt.title('moving image')
    plt.axis('off')

    plt.show()

# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space.


def display_images_with_alpha(image_z, alpha, fixed, moving):
    img = (1.0 - alpha)*fixed[:, :, image_z] + alpha*moving[:, :, image_z]
    plt.imshow(sitk.GetArrayViewFromImage(img), cmap=plt.cm.Greys_r)
    plt.axis('off')
    plt.show()

# Callback invoked when the StartEvent happens, sets up our new data.


def start_plot():
    global metric_values, multires_iterations

    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.


def end_plot():
    global metric_values, multires_iterations

    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.


def plot_values(registration_method):
    global metric_values, multires_iterations

    metric_values.append(registration_method.GetMetricValue())
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index]
             for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.show()

# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the
# metric_values list.


def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))


def plot_3d_img_slices(img, cmap="gray"):
    imageSize = img.GetSize()
    img_data = sitk.GetArrayFromImage(img)
    plt.figure(figsize=(12, 8))
    plt.subplot(131)
    plt.imshow(img_data[int(imageSize[0]/2), :, :], cmap=cmap)
    plt.title('Axial')
    plt.subplot(132)
    plt.imshow(img_data[:, int(imageSize[1]/2), :], cmap=cmap)
    plt.title('Coronal')
    plt.subplot(133)
    plt.imshow(img_data[:, :, int(imageSize[2]/2)], cmap=cmap)
    plt.title('Sagittal')
    plt.show()


def plot_3d_img_masked(img, mask, alpha=0.3):
    
    imageSize = img.GetSize()
    img_data = sitk.GetArrayFromImage(img)
    mask_data = sitk.GetArrayFromImage(mask)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(131)
    plt.imshow(img_data[int(imageSize[0]/2), :, :], cmap="gray")
    plt.imshow(mask_data[int(imageSize[0]/2), :, :],
               cmap="viridis", alpha=alpha)
    plt.title('Axial')

    plt.subplot(132)
    plt.imshow(img_data[:, int(imageSize[1]/2), :], cmap="gray")
    plt.imshow(mask_data[:, int(imageSize[1]/2), :],
               cmap="viridis", alpha=alpha)
    plt.title('Coronal')
    
    plt.subplot(133)
    plt.imshow(img_data[:, :, int(imageSize[2]/2)], cmap="gray")
    plt.imshow(mask_data[:, :, int(imageSize[2]/2)],
               cmap="viridis", alpha=alpha)
    plt.title('Sagittal')
    plt.show()

