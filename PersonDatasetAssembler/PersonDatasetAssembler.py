#!/bin/python3

# PersonDatasetAssembler - A tool to assmble images of a specific person from a
# directory of images or from a video file
# Copyright (C) 2024 Carl Philipp Klemm
#
# This file is part of PersonDatasetAssembler.
#
# PersonDatasetAssembler is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PersonDatasetAssembler is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PersonDatasetAssembler.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import os
from typing import Iterator
import cv2
import numpy
from tqdm import tqdm
from wand.exceptions import BlobError
from wand.image import Image

image_ext_ocv = [".bmp", ".jpeg", ".jpg", ".png"]
image_ext_wand = [".dng", ".arw"]


class LoadException(Exception):
	pass


def find_image_files(path: str) -> list[str]:
	paths = list()
	for root, dirs, files in os.walk(path):
		for filename in files:
			name, extension = os.path.splitext(filename)
			if extension.lower() in image_ext_ocv or extension in image_ext_wand:
				paths.append(os.path.join(root, filename))
	return paths


def image_loader(paths: list[str]) -> Iterator[numpy.ndarray]:
	for path in paths:
		name, extension = os.path.splitext(path)
		extension = extension.lower()
		if extension in image_ext_ocv:
			image = cv2.imread(path)
			if image is None:
				print(f"Warning: could not load {path}")
			else:
				yield image
		elif extension in image_ext_wand:
			try:
				image = Image(filename=path)
			except BlobError as e:
				print(f"Warning: could not load {path}, {e}")
				continue


def extract_video_images(video: cv2.VideoCapture, interval: int = 0):
	ret = True
	frame_counter = 0
	while ret:
		video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
		ret, frame = video.read()
		if ret:
			yield frame
		frame_counter += interval


def contains_face_match(detector: cv2.FaceDetectorYN, recognizer: cv2.FaceRecognizerSF, image: numpy.ndarray, referance_features: list(), thresh: float) -> bool:
	detector.setInputSize([image.shape[1], image.shape[0]])
	faces = detector.detect(image)[1]
	if faces is None:
		return 0, False
	for face in faces:
		cropped_image = recognizer.alignCrop(image, face)
		features = recognizer.feature(cropped_image)
		score_accum = 0.0
		for referance in referance_features:
			score_accum += recognizer.match(referance, features, 0)
		score = score_accum / len(referance_features)
		if score > thresh:
			return score, True
	return 0, False


def process_referance(detector: cv2.FaceDetectorYN, recognizer: cv2.FaceRecognizerSF, referance_path: str) -> list():
	images = list()
	out = list()

	if os.path.isfile(referance_path):
		image = cv2.imread(referance_path)
		if image is None:
			print(f"Could not load image from {referance_path}")
		else:
			images.append(image)
	elif os.path.isdir(referance_path):
		filenames = find_image_files(referance_path)
		images = list(image_loader(filenames))

	for image in images:
		detector.setInputSize([image.shape[1], image.shape[0]])
		faces = detector.detect(image)[1]
		if faces is None:
			print("unable to find face in referance image")
			exit(1)
		image = recognizer.alignCrop(image, faces[0])
		features = recognizer.feature(image)
		out.append(features)

	return out

if __name__ == "__main__":
	parser = argparse.ArgumentParser("Script to assemble a dataset of images of a specific person")
	parser.add_argument('--out', '-o', default="out", help="place to put dataset")
	parser.add_argument('--input', '-i', required=True, help="directory or video file to get images from")
	parser.add_argument('--skip', '-s', default=0, type=int, help="skip n frames between samples when grabbing from a video file")
	parser.add_argument('--referance', '-r', required=True, help="referance image or directory of images of the person to be found")
	parser.add_argument('--match_model', '-m', required=True, help="Path to the onnx recognition model to be used")
	parser.add_argument('--detect_model', '-d', required=True, help="Path to the onnx detection model to be used")
	parser.add_argument('--threshold', '-t', default=0.362, type=float, help="match threshold to use")
	parser.add_argument('--invert', '-n', action='store_true', help="output files that DONT match")
	args = parser.parse_args()

	recognizer = cv2.FaceRecognizerSF.create(model=args.match_model, config="", backend_id=cv2.dnn.DNN_BACKEND_DEFAULT , target_id=cv2.dnn.DNN_TARGET_CPU)
	detector = cv2.FaceDetectorYN.create(model=args.detect_model, config="", input_size=[320, 320],
		score_threshold=0.6, nms_threshold=0.3, top_k=5000, backend_id=cv2.dnn.DNN_BACKEND_DEFAULT, target_id=cv2.dnn.DNN_TARGET_CPU)

	referance_features = process_referance(detector, recognizer, args.referance)
	if len(referance_features) < 1:
		print(f"Could not load any referance image(s) from {args.referance}")
		exit(1)

	if os.path.isfile(args.input):
		video = cv2.VideoCapture(args.input)
		if not video.isOpened():
			print(f"Unable to open {args.input} as a video file")
			exit(1)
		image_generator = extract_video_images(video, args.skip + 1)
		total_images = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) / (args.skip + 1)
	elif os.path.isdir(args.input):
		image_filenams = find_image_files(args.input)
		image_generator = image_loader(image_filenams)
		total_images = len(image_filenams)
	else:
		print(f"{args.input} is not a video file nor is it a directory")
		exit(1)

	os.makedirs(args.out, exist_ok=True)

	progress = tqdm(total=int(total_images), desc="0.00")
	counter = 0
	for image in image_generator:
		if image.shape[0] > 512:
			aspect = image.shape[0] / image.shape[1]
			resized = cv2.resize(image, (int(512 / aspect), 512), 0, 0, cv2.INTER_AREA)
		else:
			resized = image
		score, match = contains_face_match(detector, recognizer, resized, referance_features, args.threshold)
		if match and not args.invert or not match and args.invert:
			filename = f"{counter:04}.png"
			cv2.imwrite(os.path.join(args.out, filename), image)
			counter += 1
		progress.set_description(f"{score:1.2f}")
		progress.update()

