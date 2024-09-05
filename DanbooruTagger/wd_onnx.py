import onnxruntime as ort
from PIL import Image
import numpy as np
import os
import hashlib
from typing import List, Union
from pathlib import Path
import csv

from utils import download


def process_image(image: Image.Image, target_size: int) -> np.ndarray:
    canvas = Image.new("RGBA", image.size, (255, 255, 255))
    canvas.paste(image, mask=image.split()[3] if image.mode == 'RGBA' else None)
    image = canvas.convert("RGB")

    # Pad image to a square
    max_dim = max(image.size)
    pad_left = (max_dim - image.size[0]) // 2
    pad_top = (max_dim - image.size[1]) // 2
    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    # Resize
    padded_image = padded_image.resize((target_size, target_size), Image.Resampling.BICUBIC)

    # Convert to numpy array
    image_array = np.asarray(padded_image, dtype=np.float32)[..., [2, 1, 0]]

    return np.expand_dims(image_array, axis=0)


def download_model():
    """
    Download the model and tags file from the server.
    :return: the path to the model and tags file
    """

    model_url = (
        "https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/model.onnx"
    )
    tags_url = "https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/selected_tags.csv"
    model_md5 = "1fc4f456261c457a08d4b9e3379cac39"
    tags_md5 = "55c11e40f95e63ea9ac21a065a73fd0f"
    model_length = 378536310
    tags_length = 308468

    home = str(Path.home()) + "/.wd_onnx/"
    if not os.path.exists(home):
        os.mkdir(home)

    model_name = "wd.onnx"
    tags_name = "selected_tags.csv"

    model_path = home + model_name
    tags_path = home + tags_name
    if os.path.exists(model_path):
        if hashlib.md5(open(model_path, "rb").read()).hexdigest() != model_md5:
            os.remove(model_path)
            if not download(model_url, model_path, model_md5, model_length):
                raise ValueError("Model download failed")

    else:
        if not download(model_url, model_path, model_md5, model_length):
            raise ValueError("Model download failed")

    if os.path.exists(tags_path):
        if hashlib.md5(open(tags_path, "rb").read()).hexdigest() != tags_md5:
            os.remove(tags_path)
            if not download(tags_url, tags_path, tags_md5, tags_length):
                raise ValueError("Tags download failed")
    else:
        if not download(tags_url, tags_path, tags_md5, tags_length):
            raise ValueError("Tags download failed")
    return model_path, tags_path


class Wd:
    def __init__(
        self,
        mode: str = "auto",
        model_path: Union[str, None] = None,
        tags_path: Union[str, None] = None,
        threshold: Union[float, int] = 0.6,
        pin_memory: bool = False,
        batch_size: int = 1,
    ):
        """
        Initialize the DeepDanbooru class.
        :param mode: the mode of the model, "cpu", "cuda", "hip" or "auto"
        :param model_path: the path to the model file
        :param tags_path: the path to the tags file
        :param threshold: the threshold of the model
        :param pin_memory: whether to use pin memory
        :param batch_size: the batch size of the model
        """

        providers = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "hip": "ROCMExecutionProvider",
            "tensorrt": "TensorrtExecutionProvider",
            "auto": (
                "CUDAExecutionProvider"
                if "CUDAExecutionProvider" in ort.get_available_providers()
                else "ROCMExecutionProvider" if "ROCMExecutionProvider" in ort.get_available_providers()
                else "CPUExecutionProvider"
            ),
        }

        if not (isinstance(threshold, float) or isinstance(threshold, int)):
            raise TypeError("threshold must be float or int")
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be between 0 and 1")
        if mode not in providers:
            raise ValueError(
                "Mode not supported. Please choose from: cpu, gpu, tensorrt"
            )
        if providers[mode] not in ort.get_available_providers():
            raise ValueError(
                f"Your device is not supported {mode}. Please choose from: cpu"
            )
        if model_path is not None and not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found")
        if tags_path is not None and not os.path.exists(tags_path):
            raise FileNotFoundError("Tags file not found")

        if model_path is None or tags_path is None:
            model_path, tags_path = download_model()

        self.session = ort.InferenceSession(model_path, providers=[providers[mode]])
        self.tags = list()
        with open(tags_path, "r") as tagfile:
            reader = csv.DictReader(tagfile)
            for row in reader:
                self.tags.append(row)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = [output.name for output in self.session.get_outputs()]
        self.threshold = threshold
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.target_size = self.session.get_inputs()[0].shape[2]
        self.mode = mode
        self.cache = {}

    def __str__(self) -> str:
        return f"Wd(mode={self.mode}, threshold={self.threshold}, pin_memory={self.pin_memory}, batch_size={self.batch_size})"

    def __repr__(self) -> str:
        return self.__str__()

    def from_image_inference(self, image: Image.Image) -> dict:
        imagenp = process_image(image, self.target_size)
        return self.predict(imagenp)

    def from_ndarray_inferece(self, image: np.ndarray) -> dict:
        if image.shape != (1, 512, 512, 3):
            raise ValueError(f"Image must be {(1, 512, 512, 3)}")
        return self.predict(image)

    def from_file_inference(self, path: str) -> dict:
        image = Image.open(path)
        return self.from_image_inference(Image.open(path))

    def inference(self, image):
        return self.session.run(self.output_name, {self.input_name: image})[0]

    def predict(self, image):
        result = self.inference(image)
        tags = self.tags
        for tag, score in zip(tags, result[0]):
            tag['score'] = float(score)
        ratings = tags[:4]
        tags = tags[4:]

        image_tags = {}

        rating = max(ratings, key=lambda el: el['score'])
        image_tags[rating['name']] = rating['score']

        for tag in tags:
            if tag['score'] >= self.threshold:
                image_tags[tag['name']] = tag['score']
        return image_tags

    def __call__(self, image) -> Union[dict, List[dict]]:
        if isinstance(image, str):
            return self.from_file_inference(image)
        elif isinstance(image, np.ndarray):
            return self.from_ndarray_inferece(image)
        elif isinstance(image, Image.Image):
            return self.from_image_inference(image)
        else:
            raise ValueError("Image must be a file path or a numpy array or list/tuple")
