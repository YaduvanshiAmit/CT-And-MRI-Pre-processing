from numpy.lib.type_check import imag
import pydicom
from skimage import morphology
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import cv2
#import nibabel 

## Read dicom file 
def DicomRead(Input_path):
    try:
        print("Reading Dicom file from:", Input_path )
        image1 = pydicom.dcmread(Input_path)
        image = image1.pixel_array
    except:
        print("Reading Nifti from: ", Input_path )
        image1 = sitk.ReadImage(Input_path)
        image = sitk.GetArrayFromImage(image1)
        
    return image

def NiftiWrite(image,output_dir,output_name=None,OutputPixelType='Uint16'):
    """
    Saving an image in either formats Uint8, Uint16
    :param Input_path: path to dicom folder which contains dicom series
    :return: Saving an image in either formats Uint8, Uint16
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if output_name is None:
        output_name='dicom_image.nii'
    
    if OutputPixelType=='Uint16':
        cast_type = sitk.sitkInt16
        
    else:
        cast_type = sitk.sitkInt8
        
    sitk.WriteImage(sitk.Cast(image,cast_type),os.path.join(output_dir,output_name))
    return 1




def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def remout(Input_path):

    data = DicomRead(Input_path)
    
    # shift the data to positive intensity value
    data -= np.min(data)
    # Removing the outliers with a probability of occuring less than 5e-3 through histogram computation
    histo, bins = np.histogram(data.flatten(), 10)
    histo = normalize(histo)
    Bin = bins[np.min(np.where(histo < 5e-3))]
    data = np.clip(data, 0, Bin)
    print(data.min(), "Min value")
    print(data.max(), "Max Value")
    
    
    return data


def Dicom_Bias_Correct(image):
    """"
    For more information please see: https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html
    """
    maskImage = sitk.OtsuThreshold(image, 0, 1, 200)
    inputImage = sitk.Cast(image, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    numberFittingLevels = 4
    numberOfIteration = [5] 
    corrector.SetMaximumNumberOfIterations(numberOfIteration * numberFittingLevels)
    imageB = corrector.Execute(inputImage, maskImage)
    fi = sitk.GetArrayFromImage(imageB)
    #plt.imshow(fi[:,:,5], cmap="gray")
    #plt.show()

    return imageB

""""
for many stabdarization process
https://github.com/jcreinhold/intensity-normalization

"""

## Function to transfer pixel to HU
def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept

    return hu_image

## Function for Organ windowing
def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    
    return window_image





## Function for removing noise
def remove_noise(file_path,window_center,window_width,display=False):
    
    image = DicomRead(file_path)
    medical_image = sitk.GetImageFromArray(image)
    hu_image = transform_to_hu(medical_image, image)
    brain_image = window_image(hu_image,window_center , window_width)

    # morphology.dilation creates a segmentation of the image
    # If one pixel is between the origin and the edge of a square of size
    # 5x5, the pixel belongs to the same class
    
    # We can instead use a circule using: morphology.disk(2)
    # In this case the pixel belongs to the same class if it's between the origin
    # and the radius
    
    segmentation = morphology.dilation(brain_image, np.ones((5, 5)))
    labels, label_nb = ndimage.label(segmentation)
    
    label_count = np.bincount(labels.ravel().astype(np.int))
    # The size of label_count is the number of classes/segmentations found
    # We don't use the first class since it's the background
    label_count[0] = 0
    
    # We create a mask with the class with more pixels
    # In this case should be the brain
    mask = labels == label_count.argmax()
    # Improve the brain mask
    mask = morphology.dilation(mask, np.ones((5, 5)))
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))
    
    # Since the the pixels in the mask are zero's and one's
    # We can multiple the original image to only keep the brain region
    masked_image = mask * brain_image
    if display:
        plt.figure(figsize=(15, 2.5))
        plt.subplot(141)
        plt.imshow(brain_image,cmap=plt.cm.bone)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(142)
        plt.imshow(mask,cmap=plt.cm.bone)
        plt.title('Mask')
        plt.axis('off')

        plt.subplot(143)
        plt.imshow(masked_image,cmap=plt.cm.bone)
        plt.title('Final Image')
        plt.axis('off')
    
        plt.show()
    return masked_image    

def crop_image(image, display=False):
    # Create a mask with the background pixels
    mask = image == 0

    # Find the brain area
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    
    # Remove the background
    croped_image = image[top_left[0]:bottom_right[0],
                top_left[1]:bottom_right[1]]
    if display:
        plt.figure(figsize=(15, 2.5))
        plt.subplot(141)
        plt.imshow(croped_image,cmap=plt.cm.bone)
        plt.title('Crop Image')
        plt.axis('off')
        plt.show()
    return croped_image    


def add_pad(image, Size,display=False):
    height, width = image.shape
    new_height = Size[0]
    new_width = Size[1]
    final_image = np.zeros((new_height, new_width))

    pad_left = int((new_width - width) / 2)
    pad_top = int((new_height - height) / 2)
    
    # Replace the pixels with the image's pixels
    final_image[pad_top:pad_top + height, pad_left:pad_left + width] = image
    
    if display:
        plt.figure(figsize=(15, 2.5))
        plt.subplot(141)
        plt.imshow(final_image,cmap=plt.cm.bone)
        plt.title('Pad Image')
        plt.axis('off')
        plt.show()
    return final_image

def resample(input_path,image,new_spacing):
    medical_image = DicomRead(input_path)
    medical_image = sitk.GetImageFromArray(medical_image)
    
    try:
        image_thickness = medical_image.SliceThickness
    except:
        image_thickness = 1    
    print(image_thickness)
    pixel_spacing = medical_image.PixelSpacing
    print(pixel_spacing)
    
    x_pixel = float(pixel_spacing[0])
    y_pixel = float(pixel_spacing[1])
    spacing = np.array([image_thickness]+list(pixel_spacing),dtype=np.float32)
    resize_factor_spacing = spacing / new_spacing
    new_spacing = spacing/resize_factor_spacing
    
    size = np.array([x_pixel, y_pixel, float(image_thickness)])
    if len(image.shape)==2:
        image_shape = np.array([image.shape[0], image.shape[1], 1])
        new_shape = image_shape * size
        new_shape = np.round(new_shape)
        resize_factor = new_shape / image_shape
        resampled_image = ndimage.interpolation.zoom(np.expand_dims(image, axis=2), resize_factor)
    
    else:
        new_shape = image.shape * size
        new_shape = np.round(new_shape)
        resize_factor = new_shape / image.shape
        resampled_image = ndimage.interpolation.zoom(np.expand_dims(image, axis=2), resize_factor)
         

    
    
    return resampled_image , new_spacing   





