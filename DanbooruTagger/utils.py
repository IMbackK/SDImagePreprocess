import requests
import shutil
import hashlib
from tqdm import tqdm


def download(url: str, save_path: str, md5: str, length: str) -> bool:
    """
    Download a file from url to save_path.
    If the file already exists, check its md5.
    If the md5 matches, return True,if the md5 doesn't match, return False.
    :param url: the url of the file to download
    :param save_path: the path to save the file
    :param md5: the md5 of the file
    :param length: the length of the file
    :return: True if the file is downloaded successfully, False otherwise
    """

    try:
        response = requests.get(url=url, stream=True)
        with open(save_path, "wb") as f:
            with tqdm.wrapattr(
                response.raw, "read", total=length, desc="Downloading"
            ) as r_raw:
                shutil.copyfileobj(r_raw, f)
        return (
            True
            if hashlib.md5(open(save_path, "rb").read()).hexdigest() == md5
            else False
        )
    except Exception as e:
        print(e)
        return False
