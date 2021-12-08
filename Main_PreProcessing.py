#!/usr/bin/env python3
# -------------------------------------------------------------------------------
# Author: Amit Yadav
# Date:   01.12.2021
# ---------------------------------------------------------------------------
from __future__ import print_function
import argparse
import random
import sys
import os
import pydicom
import numpy as np
#import SimpleITK as sitk
from skimage import morphology
from scipy import ndimage
import os
#import matplotlib.pyplot as plt
from dicom.dicom_tools import *

def arg_def():
    # --------------------------------------------------------------------------
    # Parse the commandline 

    #--------------------------------------------------------------------------
    
    # organ_dic https://radiopaedia.org/articles/windowing-ct
    Organ_Dic = {'brain': [80, 40], 'head subdural': [215, 75], 'head stroke1': [8, 32], 'head stroke2': [40, 40],
                    'head temporal bones': [2800, 600], 'head soft tissues': [375, 40],
                    'lungs': [1500, 600], 'chest mediastinum': [350, 50], 'abdomen soft tissues': [400, 50],
                    'liver': [150, 30], 'spinal soft tissues': [250, 50], 'spinal bone': [1800, 400], 'Bone': [2000, 300],'None':[4000,0]}

    parser = argparse.ArgumentParser(
            description='Extraxting the data from dicom_folder and save the results at target folder'
        )
    parser.add_argument('--dicom_folder',type=str,
            default='Data\\dicom_folder' , help='input path'
        )
    parser.add_argument('--target_folder',type=str,
            default='C:\\Users\Joel\\Desktop\\CT and MRI Preprocessing\\Data\\target_folder' , help='input path'
        )
    parser.add_argument('--image_type', type=str, default='CT',
                            help='Image type received at the intput, options are \'MRI\', \'CT\''
        ) 
    parser.add_argument('--output_image_extension', type=str, default=['png'],
                            help='Image type requested at the output, options are \'png\', \'jpeg\' and \'npy\''
        ) 
    parser.add_argument('--WL', type=int, default=None, help='Window Level for thresholding')
    parser.add_argument('--WW', type=int, default=None, help='Window Width for thresholding')
    parser.add_argument('--Normalization_type', type=str, default='MIN_MAX',
                        help='Options are "MIN_MAX" and "STND", for CT Images"')
    parser.add_argument('--Normalization_population', type=str, default='Overall',
                        help='Options are "Overall" and "Per_image", for CT Images"')
    parser.add_argument('--Output_Pixel_Type', type=str, default='Uint8',
                        help='Options are "Uint8" and "Uint16", for CT Images"')
    parser.add_argument('--Windowing_Organ', type=str, default='brain',
                            help='Organ windowing which can be one of the' + str(Organ_Dic.keys()))  
    parser.add_argument('--Image_format', type=str, default='Nifti',
                        help='Output image format, options are \'Nifti\' or \'Dicom\' ')

    args = parser.parse_args()

    if args.image_type =='CT':
        
        print(
            'Pre-processing the CT images includes windowing followed by normalization, masking_statement, resizing, and zero padding. Images are going to be:')
        if args.WL is None:
            args.WL = Organ_Dic[args.Windowing_Organ][1]
        if args.WW is None:
            args.WW = Organ_Dic[args.Windowing_Organ][0] 
        print('clipped within the window of: [' + str(
                args.WL) + '-' + str(int(args.WW)) + ',' + str(args.WL) + '+' + str(int(args.WW)) + ']')
        print('normalized using: ', args.Normalization_type)

    else:
        
        print(
            'Pre-processing for MR images includes N4-Bias Correction, Nyul Standarization, masking_statement, space-matching followed by zero padding. Images are going to be:')
        print('processed by N4 bias correction to lose their bias field')

        print('standardized using Nyul standardization method')
    
    
    return args


def main():
    args = arg_def()
    print(args.dicom_folder)
    # Get a list of all patients
    Patient_List = [ f for f in os.listdir(args.dicom_folder) if os.path.isfile(os.path.join(args.dicom_folder,f))]

    print(len(Patient_List))

    Patient_List.sort()
    random.shuffle(Patient_List)

    if args.image_type == 'CT':
        for i, patient in enumerate(Patient_List):
                Input_path = os.path.join(args.dicom_folder, patient)
                print(Input_path)
                if os.path.isfile(Input_path):
                    
                    image_M = remove_noise(Input_path,args.WL,args.WW)
                    print('bdjbjws')





    return 0 
    
if __name__ == '__main__':
    sys.exit(main())

