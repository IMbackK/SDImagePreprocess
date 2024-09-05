from deepdanbooru_onnx import DeepDanbooru
from wd_onnx import Wd
from PIL import Image
import argparse
import cv2
import os
from multiprocessing import Process, Queue
import json
from tqdm import tqdm


image_ext_ocv = [".bmp", ".jpeg", ".jpg", ".png"]


def find_image_files(path: str) -> list[str]:
	if os.path.isdir(path):
		paths = list()
		for root, dirs, files in os.walk(path):
			for filename in files:
				name, extension = os.path.splitext(filename)
				if extension.lower() in image_ext_ocv:
					paths.append(os.path.join(root, filename))
		return paths
	else:
		name, extension = os.path.splitext(path)
		if extension.lower() in image_ext_ocv:
			return [path]
		else:
			return []


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


def danbooru_pipeline(queue: Queue, image_paths: list[str], device: int, cpu: bool):
	danbooru = DeepDanbooru("cpu" if cpu else "auto")

	for path in image_paths:
		imageprompt = ""
		tags = danbooru(path)
		for tag in tags:
			imageprompt = imageprompt + ", " + tag
		imageprompt = imageprompt[2:]

		queue.put({"file_name": path, "text": imageprompt})


def wd_pipeline(queue: Queue, image_paths: list[str], device: int, cpu: bool):
	wd = Wd("cpu" if cpu else "auto", threshold=0.3)

	for path in image_paths:
		imageprompt = ""
		tags = wd(path)
		for tag in tags:
			imageprompt = imageprompt + ", " + tag
		imageprompt = imageprompt[2:]

		queue.put({"file_name": path, "text": imageprompt})


def split_list(input_list, count):
	target_length = int(len(input_list) / count)
	for i in range(0, count - 1):
		yield input_list[i * target_length: (i + 1) * target_length]
	yield input_list[(count - 1) * target_length: len(input_list)]


def save_meta(meta_file, meta, reldir, common_description):
	meta["file_name"] = os.path.relpath(meta["file_name"], reldir)
	if common_description is not None:
		meta["text"] = common_description + ", " + meta["text"]
	meta_file.write(json.dumps(meta) + '\n')


if __name__ == "__main__":
	parser = argparse.ArgumentParser("A script to tag images via DeepDanbooru")
	parser.add_argument('--common_description', '-c', help="An optional description that will be preended to the ai generated one")
	parser.add_argument('--image_dir', '-i', help="A directory containg the images to tag or a singular image to tag")
	parser.add_argument('--wd', '-w', action="store_true", help="use wd tagger instead of DeepDanbooru")
	parser.add_argument('--cpu', action="store_true", help="force cpu usge instead of gpu")
	args = parser.parse_args()

	image_paths = find_image_files(args.image_dir)

	if len(image_paths) == 0:
		print("Unable to find any images at {args.image_dir}")
		exit(1)

	nparalell = 4 if len(image_paths) > 4 else len(image_paths)
	image_path_chunks = list(split_list(image_paths, nparalell))

	print(f"Will use {nparalell} processies to create tags for {len(image_paths)} images")

	queue = Queue()
	pipe = danbooru_pipeline if not args.wd else wd_pipeline
	processies = list()
	for i in range(0, nparalell):
		processies.append(Process(target=pipe, args=(queue, image_path_chunks[i], i, args.cpu)))
		processies[-1].start()

	progress = tqdm(desc="Generateing tags", total=len(image_paths))
	exit = False

	if len(image_paths) > 1:
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
	else:
		while not exit:
			if not queue.empty():
				meta = queue.get()
				print(meta)
				progress.update()
			exit = True
			for process in processies:
				if process.is_alive():
					exit = False
					break
		while not queue.empty():
			meta = queue.get()
			print(meta)
			progress.update()

	for process in processies:
		process.join()
