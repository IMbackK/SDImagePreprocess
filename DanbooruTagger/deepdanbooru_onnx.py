import onnxruntime as ort
from PIL import Image
import numpy as np
import os
import hashlib
from typing import List, Union
from pathlib import Path

from utils import download


def process_image(image: Image.Image) -> np.ndarray:
    """
    Convert an image to a numpy array.
    :param image: the image to convert
    :return: the numpy array
    """

    image = image.convert("RGB").resize((512, 512))
    imagenp = np.array(image).astype(np.float32) / 255
    imagenp = imagenp.transpose((2, 0, 1)).reshape(1, 3, 512, 512).transpose((0, 2, 3, 1))
    return imagenp


def download_model():
    """
    Download the model and tags file from the server.
    :return: the path to the model and tags file
    """

    model_url = (
        "https://huggingface.co/chinoll/deepdanbooru/resolve/main/deepdanbooru.onnx"
    )
    tags_url = "https://huggingface.co/chinoll/deepdanbooru/resolve/main/tags.txt"
    model_md5 = "16be4e40ebcc0b1d1915bbf31f00969f"
    tags_md5 = "a3f764de985cdeba89f1d232a4204402"
    model_length = 643993025
    tags_length = 133810

    home = str(Path.home()) + "/.deepdanbooru_onnx/"
    if not os.path.exists(home):
        os.mkdir(home)

    model_name = "deepdanbooru.onnx"
    tags_name = "tags.txt"

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


class DeepDanbooru:
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
        self.tags = [i.replace("\n", "") for i in open(tags_path, "r").readlines()]

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = [output.name for output in self.session.get_outputs()]
        self.threshold = threshold
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.mode = mode
        self.cache = {}

    def __str__(self) -> str:
        return f"DeepDanbooru(mode={self.mode}, threshold={self.threshold}, pin_memory={self.pin_memory}, batch_size={self.batch_size})"

    def __repr__(self) -> str:
        return self.__str__()

    def from_image_inference(self, image: Image.Image) -> dict:
        imagenp = process_image(image)
        return self.predict(imagenp)

    def from_ndarray_inferece(self, image: np.ndarray) -> dict:
        if image.shape != (1, 512, 512, 3):
            raise ValueError(f"Image must be {(1, 512, 512, 3)}")
        return self.predict(image)

    def from_file_inference(self, image: str) -> dict:
        return self.from_image_inference(Image.open(image))

    def inference(self, image):
        return self.session.run(self.output_name, {self.input_name: image})[0]

    def predict(self, image):
        result = self.inference(image)
        image_tag = {}
        for tag, score in zip(self.tags, result[0]):
            if score >= self.threshold:
                image_tag[tag] = score
        return image_tag

    def __call__(self, image) -> Union[dict, List[dict]]:
        if isinstance(image, str):
            return self.from_file_inference(image)
        elif isinstance(image, np.ndarray):
            return self.from_ndarray_inferece(image)
        elif isinstance(image, Image.Image):
            return self.from_image_inference(image)
        else:
            raise ValueError("Image must be a file path or a numpy array or list/tuple")
