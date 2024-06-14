import warnings
from deepdanbooru_onnx import DeepDanbooru
from PIL import Image
import argparse
import cv2
import os
from multiprocessing import Process, Queue
import json
from tqdm import tqdm


image_ext_ocv = [".bmp", ".jpeg", ".jpg", ".png"]


def find_image_files(path: str) -> list[str]:
	paths = list()
	for root, dirs, files in os.walk(path):
		for filename in files:
			name, extension = os.path.splitext(filename)
			if extension.lower() in image_ext_ocv:
				paths.append(os.path.join(root, filename))
	return paths


def image_loader(paths: list[str]):
	for path in paths:
		name, extension = os.path.splitext(path)
		extension = extension.lower()
		imagebgr = cv2.imread(path)
		image = cv2.cvtColor(imagebgr, cv2.COLOR_BGR2RGB)
		if image is None:
			print(f"Warning: could not load {path}")
		else:
			image_pil = Image.fromarray(image)
			yield image_pil, path


def pipeline(queue: Queue, image_paths: list[str], device: int):
	danbooru = DeepDanbooru()

	for path in image_paths:
		imageprompt = ""
		tags = danbooru(path)
		for tag in tags:
			imageprompt = imageprompt + ", " + tag

		queue.put({"file_name": path, "text": imageprompt})


def split_list(input_list, count):
	target_length = int(len(input_list) / count)
	for i in range(0, count - 1):
		yield input_list[i * target_length: (i + 1) * target_length]
	yield input_list[(count - 1) * target_length: len(input_list)]


def save_meta(meta_file, meta, reldir, common_description):
	meta["file_name"] = os.path.relpath(meta["file_name"], reldir)
	if common_description is not None:
		meta["text"] = common_description + meta["text"]
	meta_file.write(json.dumps(meta) + '\n')


if __name__ == "__main__":
	parser = argparse.ArgumentParser("A script to tag images via DeepDanbooru")
	parser.add_argument('--batch', '-b', default=4, type=int, help="Batch size to use for inference")
	parser.add_argument('--common_description', '-c', help="An optional description that will be preended to the ai generated one")
	parser.add_argument('--image_dir', '-i', help="A directory containg the images to tag")
	args = parser.parse_args()

	nparalell = 2

	image_paths = find_image_files(args.image_dir)
	image_path_chunks = list(split_list(image_paths, nparalell))

	print(f"Will use {nparalell} processies to create tags")

	queue = Queue()
	processies = list()
	for i in range(0, nparalell):
		processies.append(Process(target=pipeline, args=(queue, image_path_chunks[i], i)))
		processies[-1].start()

	progress = tqdm(desc="Generateing tags", total=len(image_paths))
	exit = False
	with open(os.path.join(args.image_dir, "metadata.jsonl"), mode='w') as output_file:
		while not exit:
			if not queue.empty():
				meta = queue.get()
				save_meta(output_file, meta, args.image_dir, args.common_description)
				progress.update()
			exit = True
			for process in processies:
				if process.is_alive():
					exit = False
					break

		while not queue.empty():
			meta = queue.get()
			save_meta(output_file, meta, args.image_dir, args.common_description)
			progress.update()

	for process in processies:
		process.join()

