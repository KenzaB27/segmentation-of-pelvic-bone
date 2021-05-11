import numpy as np
import SimpleITK as sitk  
from utils import *
from enum import Enum 

class Interpolater(Enum):
    LINEAR = sitk.sitkLinear
    NN = sitk.sitkNearestNeighbor
    SPLINE = sitk.sitkBSpline


class Transform():
    def __init__(self, im_ref_filename, im_mov_filename):
        super().__init__()
        self.im_ref = sitk.ReadImage(im_ref_filename)
        self.im_mov = sitk.ReadImage(im_mov_filename)

class LinearTransform(Transform):
    def __init__(self, im_ref_filename, im_mov_filename):
        super().__init__(im_ref_filename, im_mov_filename)

    def apply_transf(self, lin_xfm, interp=Interpolater.LINEAR):
        """ Apply given linear transform `lin_xfm` to `self.im_mov` and 
        return the transformed image. """
        
        resampler = sitk.ResampleImageFilter()
        # Set the reference image
        resampler.SetReferenceImage(self.im_ref)
        # Use an interpolator
        resampler.SetInterpolator(interp.value)
        # Set the transformation 'lin_xfm'
        resampler.SetTransform(lin_xfm)
        trans_im_mov = resampler.Execute(self.im_mov)

        return trans_im_mov

    def est_transf(self, fix_img_mask=None, metric='MI', interp=Interpolater.LINEAR, num_iter=100, gradient_descent_step=1, conv_min_value=1e-6):
        """ Estimate linear transform to align `self.im_mov` to `self.im_ref` and 
        return the transform parameters. """

        affine_transform = sitk.AffineTransform(3)  # 3D affine transformation
        affine_transform.SetCenter([0, 0, 0])  # Set a reference center for the registration

        # Initial alignment of the two volumes
        initial_transform = sitk.CenteredTransformInitializer(
            self.im_ref, self.im_mov, affine_transform, sitk.CenteredTransformInitializerFilter.GEOMETRY)
        # initialize the registration
        registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric settings - choose an appropriate metric to be tested
        if metric == 'MI':
            registration_method.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=50)
        elif metric == 'SSD':
            registration_method.SetMetricAsMeanSquares()
        elif metric == 'NCC':
            registration_method.SetMetricAsCorrelation()

        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)

        # Mask information
        if fix_img_mask:
            registration_method.SetMetricFixedMask(fix_img_mask)

        # interpolator
        registration_method.SetInterpolator(interp.value)

        # Gradient descent optimizer
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=gradient_descent_step, numberOfIterations=num_iter, convergenceMinimumValue=conv_min_value, convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Set the initial transformation
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Connect all of the observers so that we can perform plotting during registration.
        registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
        registration_method.AddCommand(
            sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
        registration_method.AddCommand(
            sitk.sitkIterationEvent, lambda: plot_values(registration_method))

        # perform registration
        final_transform = registration_method.Execute(sitk.Cast(self.im_ref, sitk.sitkFloat32),
                                                    sitk.Cast(self.im_mov, sitk.sitkFloat32))

        # Print transformation parameters
        print(final_transform)
        print("--------")
        print('Final metric value: {0}'.format(
            registration_method.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(
            registration_method.GetOptimizerStopConditionDescription()))
        print("--------")

        return final_transform


class NonLinearTransform(Transform):
    def __init__(self, im_ref_filename, im_mov_filename):
        super().__init__(im_ref_filename, im_mov_filename)

    def est_transf(self, fix_img_mask=None, metric='MI', interp=Interpolater.LINEAR, num_iter=100, gradient_descent_step=1, conv_min_value=1e-6, fixed_points=None, moving_points=None):
        """
        Estimate non-linear transform to align `im_mov` to `im_ref` and
        return the transform parameters.
        """
        registration_method = sitk.ImageRegistrationMethod()

        # Determine the number of BSpline control points using the physical spacing we want for the control grid.
        grid_physical_spacing = [30.0, 30.0, 30.0]  # A control point every 30mm
        image_physical_size = [size*spacing for size,
                               spacing in zip(self.im_ref.GetSize(), self.im_ref.GetSpacing())]
        mesh_size = [int(image_size/grid_spacing + 0.5)
                    for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]

        # Set Initial Transform
        initial_transform = sitk.BSplineTransformInitializer(image1=self.im_ref,
                                                            transformDomainMeshSize=mesh_size, order=3)
        registration_method.SetInitialTransform(initial_transform)

        # Similarity metric settings - choose an appropriate metric to be tested
        if metric == 'MI':
            registration_method.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=50)
        elif metric == 'SSD':
            registration_method.SetMetricAsMeanSquares()
        elif metric == 'NCC':
            registration_method.SetMetricAsCorrelation()

        # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be
        # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the
        # whole image.
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)
        if fix_img_mask:
            registration_method.SetMetricFixedMask(fix_img_mask)

        # Multi-resolution framework.
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # interpolator
        registration_method.SetInterpolator(interp.value)

        # Gradient descent optimizer
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=gradient_descent_step, numberOfIterations=num_iter, convergenceMinimumValue=conv_min_value, convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # If corresponding points in the fixed and moving image are given then we display the similarity metric
        # and the TRE during the registration.
        if fixed_points and moving_points:
            registration_method.AddCommand(
                sitk.sitkStartEvent, metric_and_reference_start_plot)
            registration_method.AddCommand(
                sitk.sitkEndEvent, metric_and_reference_end_plot)
            registration_method.AddCommand(sitk.sitkIterationEvent, lambda: metric_and_reference_plot_values(
                registration_method, fixed_points, moving_points))

        # perform registration
        final_transform = registration_method.Execute(
            sitk.Cast(self.im_ref, sitk.sitkFloat32), sitk.Cast(self.im_mov, sitk.sitkFloat32))

        print(final_transform)
        print("--------")
        print("Optimizer stop condition: {0}".format(
            registration_method.GetOptimizerStopConditionDescription()))
        print("Number of iterations: {0}".format(
            registration_method.GetOptimizerIteration()))
        print("--------")
        return final_transform

    def apply_transf(self, method='Powell'):
        """ Estimate non-linear transform to align `im_mov` to `im_ref` and 
        return the transform parameters. """
        pass
