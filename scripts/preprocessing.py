#!/usr/bin/env python3

import os
import cv2
import imutils
import numpy as np
import kagglehub
from tqdm import tqdm

def crop_img(input_image: np.array) -> np.array:
	"""Finds the extreme points on the image and crops the rectangular out of them

	Args:
		input_image (np.array): input image data

	Returns:
		np.array: new criopped image data
	"""
	gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)

	# threshold the image, then perform a series of erosions +
	# dilations to remove any small regions of noise
	thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)

	# find contours in thresholded image, then grab the largest one
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)

	# find the extreme points
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])
	ADD_PIXELS = 0

	# all of it was done to extract the borders of the image content
	# now cut the detected area from the initial image
	new_img = input_image[
		extTop[1] - ADD_PIXELS : extBot[1] + ADD_PIXELS,
		extLeft[0] - ADD_PIXELS : extRight[0] + ADD_PIXELS ].copy()
	
	return new_img

def preprocess_and_write(src_dir: str, dest_dir: str, img_name: str, image_size: int):
	"""load image, preprocess it and write to the specified directory

	Args:
		src_dir (str): source directory
		dest_dir (str): destination directory
		img_name (str): image file name
		image_size (int): size of the image. length of square in pixels
	"""	
	# load image
	image = cv2.imread(os.path.join(src_dir, img_name))
	# crop image
	cropped_image = crop_img(image)
	# resize image
	cropped_image = cv2.resize(cropped_image, (image_size, image_size))
	# create a destination dir, if it is not yet exist
	if not os.path.exists(dest_dir):
		os.makedirs(dest_dir)
	# write image
	cv2.imwrite(
		os.path.join(dest_dir, img_name),
		cropped_image)


def get_raw_data() -> str:
	"""download alheimer dataset

	Returns:
		str: path to teh downloaded data
	"""	
	return os.path.join(
		kagglehub.dataset_download(
			'aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented'),
		'combined_images')


if __name__ == '__main__':
	# where raw alheimer data located, root dir for classes
	data_dir = 'data/AlzheimersData_Split' # get_raw_data()
	# wehere the preprocessed data should be copied
	dest_dir = 'data/Preprocessed224'

	for split in ['test', 'train', 'val']:
		for cl_name in os.listdir(os.path.join(data_dir, split)):
			dst_path = os.path.join(dest_dir, split, cl_name)
			src_path = os.path.join(data_dir, split, cl_name)

			image_dir = os.listdir(src_path)
			for img in tqdm(image_dir, desc=f"Copy {cl_name} {split} images"):
				preprocess_and_write(
					src_dir = src_path,
					dest_dir = dst_path,
					img_name = img,
					image_size = 224)



