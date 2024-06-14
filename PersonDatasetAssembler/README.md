### PersonDatasetAssembler

PersonDatasetAssembler is a python script that finds images of a spcific person, specified by a referance image in a directory of images or in a video file. PersonDatasetAssembler supports also raw images.

## How to use

first create a python venv and install the required packages into it:

	$ python -m venv venv
	$ source venv/bin/activate
	$ pip install -r requirements.txt

Then run PersonDatasetAssembler for instance like so:

	$ python PersonDatasetAssembler.py --referance someperson.jpg --match_model ../Weights/face_recognition_sface_2021dec.onnx --detect_model ../Weights/face_detection_yunet_2023mar.onnx --input ~/Photos --out imagesOfSomePerson

Or to extract images from a video:

	$ python PersonDatasetAssembler.py --referance someperson.jpg --match_model ../Weights/face_recognition_sface_2021dec.onnx --detect_model ../Weights/face_detection_yunet_2023mar.onnx -i ~/SomeVideo.mkv --out imagesOfSomePerson

