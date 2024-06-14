import warnings
warnings.simplefilter(action='ignore')
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, logging
import argparse
import cv2
import torch
import os
import numpy
from typing import Iterator
from torch.multiprocessing import Process, Queue
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


def image_loader(paths: list[str]) -> Iterator[numpy.ndarray]:
	for path in paths:
		name, extension = os.path.splitext(path)
		extension = extension.lower()
		imagebgr = cv2.imread(path)
		image = cv2.cvtColor(imagebgr, cv2.COLOR_BGR2RGB)
		if image is None:
			print(f"Warning: could not load {path}")
		else:
			yield image, path


def pipeline(queue: Queue, image_paths: list[str], prompt: str, device: torch.device, model_name_or_path: str, batch_size: int):
	model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=None,
		quantization_config=BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_compute_dtype=torch.float16,
			bnb_4bit_use_double_quant=False,
			bnb_4bit_quant_type='nf4',
			), device_map=device, attn_implementation="flash_attention_2")
	processor = AutoProcessor.from_pretrained(model_name_or_path)
	image_generator = image_loader(image_paths)

	stop = False
	finished_count = 0
	while not stop:
		prompts = list()
		images = list()
		filenames = list()
		for i in range(0, batch_size):
			image, filename = next(image_generator, (None, None))
			if image is None:
				stop = True
				break

			filenames.append(filename)
			images.append(image)
			prompts.append(prompt)

		if len(images) == 0:
			break

		inputs = processor(text=prompts, images=images, return_tensors="pt").to(model.device)
		generate_ids = model.generate(**inputs, max_new_tokens=100, min_new_tokens=3, length_penalty=1.0, do_sample=False, temperature=1.0, top_k=50, top_p=1.0)
		decodes = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
		finished_count += len(images)
		for i, decoded in enumerate(decodes):
			trim = len(prompt) - len("<image>")
			queue.put({"file_name": filenames[i], "text": decoded[trim:].strip()})


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
	parser = argparse.ArgumentParser("A script to tag images via llava")
	parser.add_argument('--model', '-m', default="llava-hf/llava-1.5-13b-hf", help="model to use")
	parser.add_argument('--quantize', '-q', action='store_true', help="load quantized")
	parser.add_argument('--prompt', '-p', default="Please describe this image in 10 to 20 words.", help="Prompt to use on eatch image")
	parser.add_argument('--batch', '-b', default=4, type=int, help="Batch size to use for inference")
	parser.add_argument('--common_description', '-c', help="An optional description that will be preended to the ai generated one")
	parser.add_argument('--image_dir', '-i', required=True, help="A directory containg the images to tag")
	args = parser.parse_args()

	prompt = "USER: <image>\n" + args.prompt + "\nASSISTANT: "
	os.environ["BITSANDBYTES_NOWELCOME"] = "1"

	image_paths = find_image_files(args.image_dir)
	image_path_chunks = list(split_list(image_paths, torch.cuda.device_count()))

	print(f"Will use {torch.cuda.device_count()} processies to create tags")

	logging.set_verbosity_error()
	warnings.filterwarnings("ignore")
	torch.multiprocessing.set_start_method('spawn')

	queue = Queue()
	processies = list()
	for i in range(0, torch.cuda.device_count()):
		processies.append(Process(target=pipeline, args=(queue, image_path_chunks[i], prompt, torch.device(i), args.model, args.batch)))
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

