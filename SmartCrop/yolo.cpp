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

#include <opencv2/dnn/dnn.hpp>
#include <algorithm>
#include <string>
#include <stdexcept>

#include "yolo.h"
#include "readfile.h"
#include "tokenize.h"
#include "log.h"

#define INCBIN_PREFIX r
#include "incbin.h"

INCTXT(defaultClasses, WEIGHT_DIR "/classes.txt");
INCBIN(defaultModel, WEIGHT_DIR "/yolov8x.onnx");

Yolo::Yolo(const std::filesystem::path &onnxModelPath, const cv::Size &modelInputShape,
		const std::filesystem::path& classesTxtFilePath, bool runWithOCl)
{
	modelPath = onnxModelPath;
	modelShape = modelInputShape;

	if(classesTxtFilePath.empty())
	{
		Log(Log::INFO)<<"Using builtin classes";
		loadClasses(rdefaultClassesData);
	}
	else
	{
		std::string classesStr = readFile(classesTxtFilePath);
		loadClasses(classesStr);
	}

	if(!modelPath.empty())
	{
		net = cv::dnn::readNetFromONNX(modelPath);
	}
	else
	{
		Log(Log::INFO)<<"Using builtin yolo model";
		net = cv::dnn::readNetFromONNX((const char*)rdefaultModelData, rdefaultModelSize);
	}
	if(runWithOCl)
	{
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
	}
	else
	{
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
}

std::vector<Yolo::Detection> Yolo::runInference(const cv::Mat &input)
{
	cv::Mat modelInput = input;
	if (letterBoxForSquare && modelShape.width == modelShape.height)
		modelInput = formatToSquare(modelInput);

	cv::Mat blob;
	cv::dnn::blobFromImage(modelInput, blob, 1.0/255.0, modelShape, cv::Scalar(), true, false);
	net.setInput(blob);

	std::vector<cv::Mat> outputs;
	net.forward(outputs, net.getUnconnectedOutLayersNames());

	int rows = outputs[0].size[1];
	int dimensions = outputs[0].size[2];

	bool yolov8 = false;
	// yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
	// yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
	if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
	{
		yolov8 = true;
		rows = outputs[0].size[2];
		dimensions = outputs[0].size[1];

		outputs[0] = outputs[0].reshape(1, dimensions);
		cv::transpose(outputs[0], outputs[0]);
	}
	float *data = (float *)outputs[0].data;

	float x_factor = modelInput.cols / modelShape.width;
	float y_factor = modelInput.rows / modelShape.height;

	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	for (int i = 0; i < rows; ++i)
	{
		if (yolov8)
		{
			float *classes_scores = data+4;

			cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
			cv::Point class_id;
			double maxClassScore;

			minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

			if (maxClassScore > modelScoreThreshold)
			{
				confidences.push_back(maxClassScore);
				class_ids.push_back(class_id.x);

				float x = data[0];
				float y = data[1];
				float w = data[2];
				float h = data[3];

				int left = int((x - 0.5 * w) * x_factor);
				int top = int((y - 0.5 * h) * y_factor);

				int width = int(w * x_factor);
				int height = int(h * y_factor);

				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
		else // yolov5
		{
			float confidence = data[4];

			if (confidence >= modelConfidenceThreshold)
			{
				float *classes_scores = data+5;

				cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
				cv::Point class_id;
				double max_class_score;

				minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

				if (max_class_score > modelScoreThreshold)
				{
					confidences.push_back(confidence);
					class_ids.push_back(class_id.x);

					float x = data[0];
					float y = data[1];
					float w = data[2];
					float h = data[3];

					int left = int((x - 0.5 * w) * x_factor);
					int top = int((y - 0.5 * h) * y_factor);

					int width = int(w * x_factor);
					int height = int(h * y_factor);

					boxes.push_back(cv::Rect(left, top, width, height));
				}
			}
		}

		data += dimensions;
	}

	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

	std::vector<Yolo::Detection> detections{};
	for(unsigned long i = 0; i < nms_result.size(); ++i)
	{
		int idx = nms_result[i];

		Yolo::Detection result;
		result.class_id = class_ids[idx];
		result.confidence = confidences[idx];

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dis(100, 255);
		result.color = cv::Scalar(dis(gen),
		                          dis(gen),
		                          dis(gen));

		result.className = classes[result.class_id].first;
		result.priority = classes[result.class_id].second;
		clampBox(boxes[idx], input.size());
		result.box = boxes[idx];
		detections.push_back(result);
	}

	return detections;
}


void Yolo::clampBox(cv::Rect& box, const cv::Size& size)
{
	if(box.x < 0)
	{
		box.width += box.x;
		box.x = 0;
	}
	if(box.y < 0)
	{
		box.height += box.y;
		box.y = 0;
	}
	if(box.x+box.width > size.width)
		box.width = size.width - box.x;
	if(box.y+box.height > size.height)
		box.height = size.height - box.y;
}

void Yolo::loadClasses(const std::string& classesStr)
{
	std::vector<std::string> candidateClasses = tokenizeBinaryIgnore(classesStr, '\n', '"', '\\');
	classes.clear();
	for(std::string& instance : candidateClasses)
	{
		if(instance.size() < 2)
			continue;

		std::vector<std::string> tokens = tokenizeBinaryIgnore(instance, ',', '"', '\\');

		if(*tokens[0].begin() == '"')
			instance.erase(tokens[0].begin());
		if(tokens[0].back() == '"')
			tokens[0].pop_back();
		int priority = -1;
		if(tokens.size() > 1)
		{
			try
			{
				priority = std::stoi(tokens[1]);
			}
			catch(const std::invalid_argument& err)
			{
				Log(Log::WARN)<<"unable to get priority for class "<<tokens[0]<<' '<<err.what();
			}
		}
		classes.push_back({tokens[0], priority});
	}
}

cv::Mat Yolo::formatToSquare(const cv::Mat &source)
{
	int col = source.cols;
	int row = source.rows;
	int _max = MAX(col, row);
	cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
	source.copyTo(result(cv::Rect(0, 0, col, row)));
	return result;
}

int Yolo::getClassForStr(const std::string& str) const
{
	for(size_t i = 0; i < classes.size(); ++i)
	{
		if(classes[i].first == str)
			return i;
	}
	return -1;
}
