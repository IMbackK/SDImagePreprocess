//
// SmartCrop - A tool for content aware croping of images
// Copyright (C) 2024 Carl Philipp Klemm
//
// This file is part of SmartCrop.
//
// SmartCrop is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// SmartCrop is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with SmartCrop.  If not, see <http://www.gnu.org/licenses/>.
//

#include "facerecognizer.h"
#include <filesystem>

#define INCBIN_PREFIX r
#include "incbin.h"

INCBIN(defaultRecognizer, WEIGHT_DIR "/face_recognition_sface_2021dec.onnx");
INCBIN(defaultDetector, WEIGHT_DIR "/face_detection_yunet_2023mar.onnx");

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>

#include "log.h"

static const std::vector<unsigned char> onnx((unsigned char*)rdefaultDetectorData, ((unsigned char*)rdefaultDetectorData)+rdefaultDetectorSize);

FaceRecognizer::FaceRecognizer(std::filesystem::path recognizerPath, const std::filesystem::path& detectorPath, const std::vector<cv::Mat>& referances)
{
	if(detectorPath.empty())
	{
		Log(Log::INFO)<<"Using builtin face detection model";

		detector = cv::FaceDetectorYN::create("onnx", onnx, std::vector<unsigned char>(), {320, 320}, 0.6, 0.3, 5000, cv::dnn::Backend::DNN_BACKEND_OPENCV, cv::dnn::Target::DNN_TARGET_CPU);
		if(!detector)
			throw LoadException("Unable to load detector network from built in file");
	}
	else
	{
		detector = cv::FaceDetectorYN::create(detectorPath, "", {320, 320}, 0.6, 0.3, 5000, cv::dnn::Backend::DNN_BACKEND_OPENCV, cv::dnn::Target::DNN_TARGET_CPU);
		if(!detector)
			throw LoadException("Unable to load detector network from "+detectorPath.string());
	}

	bool defaultNetwork = recognizerPath.empty();

	if(defaultNetwork)
	{
		Log(Log::INFO)<<"Using builtin face recognition model";
		recognizerPath = cv::tempfile("onnx");
		std::ofstream file(recognizerPath);
		if(!file.is_open())
			throw LoadException("Unable open temporary file at "+recognizerPath.string());
		Log(Log::DEBUG)<<"Using "<<recognizerPath<<" as temporary file for onnx recongnition network";
		file.write(reinterpret_cast<const char*>(rdefaultRecognizerData), rdefaultRecognizerSize);
		file.close();
	}

	recognizer = cv::FaceRecognizerSF::create(recognizerPath.string(), "", cv::dnn::Backend::DNN_BACKEND_OPENCV, cv::dnn::Target::DNN_TARGET_CPU);

	if(defaultNetwork)
		std::filesystem::remove(recognizerPath);

	if(!recognizer)
		throw LoadException("Unable to load recognizer network from "+recognizerPath.string());

	addReferances(referances);
}

cv::Mat FaceRecognizer::detectFaces(const cv::Mat& input)
{
	detector->setInputSize(input.size());
	cv::Mat faces;
	detector->detect(input, faces);
	return faces;
}

bool FaceRecognizer::addReferances(const std::vector<cv::Mat>& referances)
{
	bool ret = false;
	for(const cv::Mat& image : referances)
	{
		cv::Mat faces = detectFaces(image);
		assert(faces.cols == 15);
		if(faces.empty())
		{
			Log(Log::WARN)<<"A referance image provided dose not contian any face";
			continue;
		}
		if(faces.rows > 1)
			Log(Log::WARN)<<"A referance image provided contains more than one face, only the first detected face will be considered";
		cv::Mat cropedImage;
		recognizer->alignCrop(image, faces.row(0), cropedImage);
		cv::Mat features;
		recognizer->feature(cropedImage, features);
		referanceFeatures.push_back(features.clone());
		ret = true;
	}

	return ret;
}

void FaceRecognizer::setThreshold(double threasholdIn)
{
	threshold = threasholdIn;
}

double FaceRecognizer::getThreshold()
{
	return threshold;
}

void FaceRecognizer::clearReferances()
{
	referanceFeatures.clear();
}

FaceRecognizer::Detection FaceRecognizer::isMatch(const cv::Mat& input, bool alone)
{
	cv::Mat faces = detectFaces(input);

	Detection bestMatch;
	bestMatch.confidence = 0;
	bestMatch.person = -1;

	if(alone && faces.rows > 1)
	{
		bestMatch.person = -2;
		return bestMatch;
	}

	for(int i = 0; i < faces.rows; ++i)
	{
		cv::Mat face;
		recognizer->alignCrop(input, faces.row(i), face);
		cv::Mat features;
		recognizer->feature(face, features);
		features = features.clone();
		for(size_t referanceIndex = 0; referanceIndex < referanceFeatures.size(); ++referanceIndex)
		{
			double score = recognizer->match(referanceFeatures[referanceIndex], features, cv::FaceRecognizerSF::FR_COSINE);
			if(score > threshold && score > bestMatch.confidence)
			{
				bestMatch.confidence = score;
				bestMatch.person = referanceIndex;
				bestMatch.rect = cv::Rect(faces.at<int>(i, 0), faces.at<int>(i, 1), faces.at<int>(i, 2), faces.at<int>(i, 3));
			}
		}
	}

	return bestMatch;
}
