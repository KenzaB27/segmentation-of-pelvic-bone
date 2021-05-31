from IPython.display import clear_output
import matplotlib.pyplot as plt
from scipy import linalg
import SimpleITK as sitk
import numpy as np


def save_image(image, fileName):
    writer = sitk.ImageFileWriter()
    writer.SetImageIO("NiftiImageIO")
    writer.SetFileName(fileName)
    writer.Execute(image)
    return True


def read_image(file_name):
    img = sitk.ReadImage(file_name)
    img = sitk.Cast(img, sitk.sitkFloat32)
    return img


def plot_3d_img_slices(img, cmap="gray"):
    img_data = sitk.GetArrayFromImage(img)
    imageSize = img_data.shape
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

    #imageSize = img.GetSize()
    img_data = sitk.GetArrayFromImage(img)
    mask_data = sitk.GetArrayFromImage(mask)
    imageSize = img_data.shape

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


def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    """
    Callback invoked by the interact IPython method for scrolling through the image stacks of
    the two images (moving and fixed).
    """
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


def display_images_with_alpha(image_z, alpha, fixed, moving):
    """ 
    Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
    of an image stack of two images that occupy the same physical space.
    """
    img = (1.0 - alpha)*fixed[:, :, image_z] + alpha*moving[:, :, image_z]
    plt.imshow(sitk.GetArrayViewFromImage(img), cmap=plt.cm.Greys_r)
    plt.axis('off')
    plt.show()


def start_plot():
    """ Callback invoked when the StartEvent happens, sets up our new data. """
    global metric_values, multires_iterations

    metric_values = []
    multires_iterations = []


def end_plot():
    """Callback invoked when the EndEvent happens, do cleanup of data and figure."""
    global metric_values, multires_iterations

    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()


def plot_values(registration_method):
    """Callback invoked when the IterationEvent happens, update our data and display new figure."""
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


def update_multires_iterations():
    """
    Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the
    metric_values list.
    """
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))


def metric_update_multires_iterations():
    """
    Callback we associate with the MultiResolutionIterationEvent, update the
    index into the metric_values list.
    """
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))


def metric_and_reference_start_plot():
    """ Callback we associate with the StartEvent, sets up our new data."""
    global metric_values, multires_iterations, reference_mean_values
    global reference_min_values, reference_max_values
    global current_iteration_number

    metric_values = []
    multires_iterations = []
    reference_mean_values = []
    reference_min_values = []
    reference_max_values = []
    current_iteration_number = -1


def metric_and_reference_end_plot():
    """Callback we associate with the EndEvent, do cleanup of data and figure."""
    global metric_values, multires_iterations, reference_mean_values
    global reference_min_values, reference_max_values
    global current_iteration_number

    del metric_values
    del multires_iterations
    del reference_mean_values
    del reference_min_values
    del reference_max_values
    del current_iteration_number
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()


def metric_and_reference_plot_values(registration_method, fixed_points, moving_points):
    """
    Callback we associate with the IterationEvent, update our data and display
    new figure.
    """
    global metric_values, multires_iterations, reference_mean_values
    global reference_min_values, reference_max_values
    global current_iteration_number

    # Some optimizers report an iteration event for function evaluations and not
    # a complete iteration, we only want to update every iteration.
    if registration_method.GetOptimizerIteration() == current_iteration_number:
        return

    current_iteration_number = registration_method.GetOptimizerIteration()
    metric_values.append(registration_method.GetMetricValue())
    # Compute and store TRE statistics (mean, min, max).
    current_transform = sitk.CompositeTransform(
        registration_method.GetInitialTransform())
    current_transform.SetParameters(registration_method.GetOptimizerPosition())
    current_transform.AddTransform(
        registration_method.GetMovingInitialTransform())
    current_transform.AddTransform(
        registration_method.GetFixedInitialTransform().GetInverse())
    mean_error, _, min_error, max_error, _ = registration_errors(
        current_transform, fixed_points, moving_points)
    reference_mean_values.append(mean_error)
    reference_min_values.append(min_error)
    reference_max_values.append(max_error)

    # Clear the output area (wait=True, to reduce flickering), and plot current data.
    clear_output(wait=True)
    # Plot the similarity metric values.
    plt.subplot(1, 2, 1)
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index]
                                   for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    # Plot the TRE mean value and the [min-max] range.
    plt.subplot(1, 2, 2)
    plt.plot(reference_mean_values, color='black', label='mean')
    plt.fill_between(range(len(reference_mean_values)), reference_min_values, reference_max_values,
                     facecolor='red', alpha=0.5)
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('TRE [mm]', fontsize=12)
    plt.legend()

    # Adjust the spacing between subplots so that the axis labels don't overlap.
    plt.tight_layout()
    plt.show()


def registration_errors(tx, reference_fixed_point_list, reference_moving_point_list,
                        display_errors=False, min_err=None, max_err=None, figure_size=(8, 6)):
    """
    Distances between points transformed by the given transformation and their
    location in another coordinate system. When the points are only used to 
    evaluate registration accuracy (not used in the registration) this is the 
    Target Registration Error (TRE).

    Args:
        tx (SimpleITK.Transform): The transform we want to evaluate.
        reference_fixed_point_list (list(tuple-like)): Points in fixed image 
                                                       cooredinate system.
        reference_moving_point_list (list(tuple-like)): Points in moving image 
                                                        cooredinate system.
        display_errors (boolean): Display a 3D figure with the points from 
                                  reference_fixed_point_list color corresponding
                                  to the error.
        min_err, max_err (float): color range is linearly stretched between min_err 
                                  and max_err. If these values are not given then
                                  the range of errors computed from the data is used.
        figure_size (tuple): Figure size in inches.
    Returns:
     (mean, std, min, max, errors) (float, float, float, float, [float]): 
      TRE statistics and original TREs.
    """
    transformed_fixed_point_list = [tx.TransformPoint(
        p) for p in reference_fixed_point_list]

    errors = [linalg.norm(np.array(p_fixed) - np.array(p_moving))
              for p_fixed, p_moving in zip(transformed_fixed_point_list, reference_moving_point_list)]
    min_errors = np.min(errors)
    max_errors = np.max(errors)
    if display_errors:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        import matplotlib
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111, projection='3d')
        if not min_err:
            min_err = min_errors
        if not max_err:
            max_err = max_errors

        collection = ax.scatter(list(np.array(reference_fixed_point_list).T)[0],
                                list(np.array(reference_fixed_point_list).T)[
            1],
            list(np.array(reference_fixed_point_list).T)[
            2],
            marker='o',
            c=errors,
            vmin=min_err,
            vmax=max_err,
            cmap=matplotlib.cm.hot,
            label='fixed points')
        plt.colorbar(collection, shrink=0.8)
        plt.title('registration errors in mm', x=0.7, y=1.05)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    return (np.mean(errors), np.std(errors), min_errors, max_errors, errors)


def create_ref_domain(images):
    dimension = images[0].GetDimension()

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)
    for img in images:
        reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                      zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

    # Create the reference image with a zero origin, identity direction cosine matrix and dimension
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()

    # Select arbitrary number of pixels per dimension, smallest size that yields desired results
    # or the required size of a pretrained network (e.g. VGG-16 224x224), transfer learning. This will
    # often result in non-isotropic pixel spacing.
    reference_size = [256, 256, 128]  # * dimension
    reference_spacing = [
        phys_sz / (sz - 1) for sz, phys_sz in zip(reference_size, reference_physical_size)]

    # Note, if we get problems with the masks then maybe this is a good idea!

    # Another possibility is that you want isotropic pixels, then you can specify the image size for one of
    # the axes and the others are determined by this choice. Below we choose to set the x axis to 128 and the
    # spacing set accordingly.
    # Uncomment the following lines to use this strategy.
    # reference_size_x = 128
    # reference_spacing = [reference_physical_size[0]/(reference_size_x-1)]*dimension
    # reference_size = [int(phys_sz/(spc) + 1) for phys_sz,spc in zip(reference_physical_size, reference_spacing)]

    reference_image = sitk.Image(reference_size, images[0].GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as
    # this takes into account size, spacing and direction cosines. For the vast majority of images the direction
    # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the
    # spacing will not yield the correct coordinates resulting in a long debugging session.
    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    return [reference_image, reference_center]


def resample_all_images(images1, masks1):
    images = []
    masks = []
    for img, mask in zip(images1, masks1):
        [image, mask] = threshold_based_crop(img, mask)
        images.append(image)
        masks.append(mask)

    [reference_image, reference_center] = create_ref_domain(images)
    resampled_images = []
    resampled_masks = []
    for image, mask in zip(images, masks):
        transform = sitk.CenteredTransformInitializer(reference_image,
                                                      image,
                                                      sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
        resampled_images.append(sitk.Resample(
            image, reference_image, transform, sitk.sitkLinear, 0.0, image.GetPixelID()))
        resampled_masks.append(sitk.Resample(
            mask, reference_image, transform, sitk.sitkNearestNeighbor, 0.0, image.GetPixelID()))
    return [resampled_images, resampled_masks]


def threshold_based_crop(image, mask):
    '''
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. Then crop the image using the foreground's axis aligned bounding box.
    Args:
        image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Return:
        Cropped image based on foreground's axis aligned bounding box.
    '''
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    inside_value = 0
    outside_value = 255
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(sitk.OtsuThreshold(
        image, inside_value, outside_value))
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    image_output = sitk.RegionOfInterest(image, bounding_box[int(
        len(bounding_box) / 2):], bounding_box[0:int(len(bounding_box) / 2)])
    mask_output = sitk.RegionOfInterest(mask, bounding_box[int(
        len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])
    return image_output, mask_output
