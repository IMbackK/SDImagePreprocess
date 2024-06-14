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

#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <execution>
#include <string>
#include <vector>
#include <numeric>

#include "yolo.h"
#include "log.h"
#include "options.h"
#include "utils.h"
#include "intelligentroi.h"
#include "seamcarving.h"
#include "facerecognizer.h"

const Yolo::Detection* pointInDetectionHoriz(int x, const std::vector<Yolo::Detection>& detections, const Yolo::Detection* ignore = nullptr)
{
	const Yolo::Detection* inDetection = nullptr;
	for(const Yolo::Detection& detection : detections)
	{
		if(ignore && ignore == &detection)
			continue;

		if(detection.box.x <= x && detection.box.x+detection.box.width >= x)
		{
			if(!inDetection || detection.box.br().x > inDetection->box.br().x)
			inDetection = &detection;
		}
	}
	return inDetection;
}

bool findRegionEndpointHoriz(int& x, const std::vector<Yolo::Detection>& detections, int imgSizeX)
{
	const Yolo::Detection* inDetection = pointInDetectionHoriz(x, detections);

	Log(Log::DEBUG, false)<<__func__<<" point "<<x;

	if(!inDetection)
	{
		const Yolo::Detection* closest = nullptr;
		for(const Yolo::Detection& detection : detections)
		{
			if(detection.box.x > x)
			{
				if(closest == nullptr || detection.box.x-x > closest->box.x-x)
					closest = &detection;
			}
		}
		if(closest)
			x = closest->box.x;
		else
			x = imgSizeX;

		Log(Log::DEBUG)<<" is not in any box and will be moved to "<<x<<" where the closest box ("<<(closest ? closest->className : "null")<<") is";
		return false;
	}
	else
	{
		x = inDetection->box.br().x;
		Log(Log::DEBUG, false)<<" is in a box and will be moved to its end "<<x<<" where ";
		const Yolo::Detection* candidateDetection = pointInDetectionHoriz(x, detections, inDetection);
		if(candidateDetection && candidateDetection->box.br().x > x)
		{
			Log(Log::DEBUG)<<"it is again in a box";
			return findRegionEndpointHoriz(x, detections, imgSizeX);
		}
		else
		{
			Log(Log::DEBUG)<<"it is not in a box";
			return true;
		}
	}
}

std::vector<std::pair<cv::Mat, bool>> cutImageIntoHorzRegions(cv::Mat& image, const std::vector<Yolo::Detection>& detections)
{
	std::vector<std::pair<cv::Mat, bool>> out;

	std::cout<<__func__<<' '<<image.cols<<'x'<<image.rows<<std::endl;

	for(int x = 0; x < image.cols; ++x)
	{
		int start = x;
		bool frozen = findRegionEndpointHoriz(x, detections, image.cols);

		int width = x-start;
		if(x < image.cols)
			++width;
		cv::Rect rect(start, 0, width, image.rows);
		Log(Log::DEBUG)<<__func__<<" region\t"<<rect;
		cv::Mat slice = image(rect);
		out.push_back({slice, frozen});
	}

	return out;
}

cv::Mat assembleFromSlicesHoriz(const std::vector<std::pair<cv::Mat, bool>>& slices)
{
	assert(!slices.empty());

	int cols = 0;
	for(const std::pair<cv::Mat, bool>& slice : slices)
		cols += slice.first.cols;


	cv::Mat image(cols, slices[0].first.rows, slices[0].first.type());
	Log(Log::DEBUG)<<__func__<<' '<<image.size()<<' '<<cols<<' '<<slices[0].first.rows;

	int col = 0;
	for(const std::pair<cv::Mat, bool>& slice : slices)
	{
		cv::Rect rect(col, 0, slice.first.cols, slice.first.rows);
		Log(Log::DEBUG)<<__func__<<' '<<rect;
		slice.first.copyTo(image(rect));
		col += slice.first.cols-1;
	}

	return image;
}

void transposeRect(cv::Rect& rect)
{
	int x = rect.x;
	rect.x = rect.y;
	rect.y = x;

	int width = rect.width;
	rect.width = rect.height;
	rect.height = width;
}

bool seamCarveResize(cv::Mat& image, std::vector<Yolo::Detection> detections, double targetAspectRatio = 1.0)
{
	detections.erase(std::remove_if(detections.begin(), detections.end(), [](const Yolo::Detection& detection){return detection.priority < 3;}), detections.end());

	double aspectRatio = image.cols/static_cast<double>(image.rows);

	Log(Log::DEBUG)<<"Image size "<<image.size()<<" aspect ratio "<<aspectRatio<<" target aspect ratio "<<targetAspectRatio;

	bool vertical = false;
	if(aspectRatio > targetAspectRatio)
		vertical = true;

	int requiredLines = 0;
	if(!vertical)
		requiredLines = image.rows*targetAspectRatio - image.cols;
	else
		requiredLines = image.cols/targetAspectRatio - image.rows;

	Log(Log::DEBUG)<<__func__<<' '<<requiredLines<<" lines are required in "<<(vertical ? "vertical" : "horizontal")<<" direction";

	if(vertical)
	{
		cv::transpose(image, image);
		for(Yolo::Detection& detection : detections)
			transposeRect(detection.box);
	}

	std::vector<std::pair<cv::Mat, bool>> slices = cutImageIntoHorzRegions(image, detections);
	Log(Log::DEBUG)<<"Image has "<<slices.size()<<" slices:";
	int totalResizableSize = 0;
	for(const std::pair<cv::Mat, bool>& slice : slices)
	{
		Log(Log::DEBUG)<<"a "<<(slice.second ? "frozen" : "unfrozen")<<" slice of size "<<slice.first.cols;
		if(!slice.second)
			totalResizableSize += slice.first.cols;
	}

	if(totalResizableSize < requiredLines+1)
	{
		Log(Log::WARN)<<"Unable to seam carve as there are only "<<totalResizableSize<<" unfrozen cols";
		if(vertical)
			cv::transpose(image, image);
		return false;
	}

	std::vector<int> seamsForSlice(slices.size(), 0);
	for(size_t i = 0; i < slices.size(); ++i)
	{
		if(!slices[i].second)
			seamsForSlice[i] = (static_cast<double>(slices[i].first.cols)/totalResizableSize)*requiredLines;
	}

	int residual = requiredLines - std::accumulate(seamsForSlice.begin(), seamsForSlice.end(), decltype(seamsForSlice)::value_type(0));;
	for(ssize_t i = slices.size()-1; i >= 0; --i)
	{
		if(!slices[i].second)
		{
			seamsForSlice[i] += residual;
			break;
		}
	}

	for(size_t i = 0; i < slices.size(); ++i)
	{
		if(seamsForSlice[i] != 0)
		{
			bool ret = SeamCarving::strechImage(slices[i].first, seamsForSlice[i], true);
			if(!ret)
			{
				if(vertical)
					transpose(image, image);
				return false;
			}
		}
	}

	image = assembleFromSlicesHoriz(slices);

	if(vertical)
		cv::transpose(image, image);

	return true;
}

void drawDebugInfo(cv::Mat &image, const cv::Rect& rect, const std::vector<Yolo::Detection>& detections)
{
	for(const Yolo::Detection& detection : detections)
	{
		cv::rectangle(image, detection.box, detection.color, 3);
		std::string label = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4) + ' ' + std::to_string(detection.priority);
		cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 1, 1, 0);
		cv::Rect textBox(detection.box.x, detection.box.y - 40, labelSize.width + 10, labelSize.height + 20);
		cv::rectangle(image, textBox, detection.color, cv::FILLED);
		cv::putText(image, label, cv::Point(detection.box.x + 5, detection.box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 1, 0);
	}

	cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 8);
}

static void reduceSize(cv::Mat& image, const cv::Size& targetSize)
{
	int longTargetSize = std::max(targetSize.width, targetSize.height)*2;
	if(std::max(image.cols, image.rows) > longTargetSize)
	{
		if(image.cols > image.rows)
		{
			double ratio = static_cast<double>(longTargetSize)/image.cols;
			cv::resize(image, image, {longTargetSize, static_cast<int>(image.rows*ratio)}, 0, 0, ratio < 1 ? cv::INTER_AREA : cv::INTER_CUBIC);
		}
		else
		{
			double ratio = static_cast<double>(longTargetSize)/image.rows;
			cv::resize(image, image, {static_cast<int>(image.cols*ratio), longTargetSize}, 0, 0, ratio < 1 ? cv::INTER_AREA : cv::INTER_CUBIC);
		}
	}
}

void pipeline(const std::filesystem::path& path, const Config& config, Yolo& yolo, FaceRecognizer* recognizer,
	std::mutex& reconizerMutex, const std::filesystem::path& debugOutputPath)
{
	InteligentRoi intRoi(yolo);
	cv::Mat image = cv::imread(path);
	if(!image.data)
	{
		Log(Log::WARN)<<"could not load image "<<path<<" skipping";
		return;
	}

	reduceSize(image, config.targetSize);

	std::vector<Yolo::Detection> detections = yolo.runInference(image);

	Log(Log::DEBUG)<<"Got "<<detections.size()<<" detections for "<<path;
	for(Yolo::Detection& detection : detections)
	{
		bool hasmatch = false;
		if(recognizer && detection.className == "person")
		{
			cv::Mat person = image(detection.box);
			reconizerMutex.lock();
			FaceRecognizer::Detection match = recognizer->isMatch(person);
			reconizerMutex.unlock();
			if(match.person >= 0)
			{
				detection.priority += 10;
				hasmatch = true;
				detections.push_back({0, "Face", match.confidence, 20, {255, 0, 0}, match.rect});
			}
		}
		Log(Log::DEBUG)<<detection.class_id<<": "<<detection.className<<" at "<<detection.box<<" with prio "<<detection.priority<<(hasmatch ? " has match" : "");
	}

	cv::Rect crop;
	bool incompleate = intRoi.getCropRectangle(crop, detections, image.size());

	if(config.seamCarving && incompleate)
	{
		bool ret = seamCarveResize(image, detections, config.targetSize.aspectRatio());
		if(ret && image.size().aspectRatio() != config.targetSize.aspectRatio())
		{
			detections = yolo.runInference(image);
		}
	}

	cv::Mat croppedImage;

	if(image.size().aspectRatio() != config.targetSize.aspectRatio() && incompleate)
	{
		intRoi.getCropRectangle(crop, detections, image.size());

		if(config.debug)
		{
			cv::Mat debugImage = image.clone();
			drawDebugInfo(debugImage, crop, detections);
			bool ret = cv::imwrite(debugOutputPath/path.filename(), debugImage);
			if(!ret)
				Log(Log::WARN)<<"could not save debug image to "<<debugOutputPath/path.filename()<<" skipping";
		}

		croppedImage = image(crop);
	}
	else if(!incompleate)
	{
		croppedImage = image(crop);
	}
	else
	{
		croppedImage = image;
	}

	cv::Mat resizedImage;
	cv::resize(croppedImage, resizedImage, config.targetSize, 0, 0, cv::INTER_CUBIC);
	bool ret = cv::imwrite(config.outputDir/path.filename(), resizedImage);
	if(!ret)
		Log(Log::WARN)<<"could not save image to "<<config.outputDir/path.filename()<<" skipping";
}

void threadFn(const std::vector<std::filesystem::path>& images, const Config& config, FaceRecognizer* recognizer,
		std::mutex& reconizerMutex, const std::filesystem::path& debugOutputPath)
{
	Yolo yolo(config.modelPath, {640, 480}, config.classesPath, false);
	for(std::filesystem::path path : images)
		pipeline(path, config, yolo, recognizer, reconizerMutex, debugOutputPath);
}

template<typename T>
std::vector<std::vector<T>> splitVector(const std::vector<T>& vec, size_t parts)
{
	std::vector<std::vector<T>> out;

	size_t length = vec.size()/parts;
	size_t remain = vec.size() % parts;

	size_t begin = 0;
	size_t end = 0;

	for (size_t i = 0; i < std::min(parts, vec.size()); ++i)
	{
		end += (remain > 0) ? (length + !!(remain--)) : length;
		out.push_back(std::vector<T>(vec.begin() + begin, vec.begin() + end));
		begin = end;
	}

	return out;
}

int main(int argc, char* argv[])
{
	Log::level = Log::INFO;

	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	if(config.outputDir.empty())
	{
		Log(Log::ERROR)<<"a output path \"-o\" is required";
		return 1;
	}

	if(config.imagePaths.empty())
	{
		Log(Log::ERROR)<<"at least one input image or directory is required";
		return 1;
	}

	std::vector<std::filesystem::path> imagePaths;

	for(const std::filesystem::path& path : config.imagePaths)
		getImageFiles(path, imagePaths);

	Log(Log::DEBUG)<<"Images:";
	for(const::std::filesystem::path& path: imagePaths)
		Log(Log::DEBUG)<<path;

	if(imagePaths.empty())
	{
		Log(Log::ERROR)<<"no image was found\n";
		return 1;
	}

	if(!std::filesystem::exists(config.outputDir))
	{
		if(!std::filesystem::create_directory(config.outputDir))
		{
			Log(Log::ERROR)<<"could not create directory at "<<config.outputDir;
			return 1;
		}
	}

	std::filesystem::path debugOutputPath(config.outputDir/"debug");
	if(config.debug)
	{
		if(!std::filesystem::exists(debugOutputPath))
			std::filesystem::create_directory(debugOutputPath);
	}

	FaceRecognizer* recognizer = nullptr;
	std::mutex recognizerMutex;
	if(!config.focusPersonImage.empty())
	{
		cv::Mat personImage = cv::imread(config.focusPersonImage);
		if(personImage.empty())
		{
			Log(Log::ERROR)<<"Could not load image from "<<config.focusPersonImage;
			return 1;
		}
		recognizer = new FaceRecognizer();
		recognizer->addReferances({personImage});
		recognizer->setThreshold(config.threshold);
	}

	std::vector<std::thread> threads;
	std::vector<std::vector<std::filesystem::path>> imagePathParts = splitVector(imagePaths, std::thread::hardware_concurrency());

	for(size_t i = 0; i < imagePathParts.size(); ++i)
		threads.push_back(std::thread(threadFn, imagePathParts[i], std::ref(config),  recognizer, std::ref(recognizerMutex), std::ref(debugOutputPath)));

	for(std::thread& thread : threads)
		thread.join();

	return 0;
}
