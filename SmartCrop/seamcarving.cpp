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

#include "seamcarving.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>
#include <cfloat>
#include <vector>
#include "log.h"

bool SeamCarving::strechImage(cv::Mat& image, int seams, bool grow, std::vector<std::vector<int>>* seamsVect)
{
	cv::Mat newFrame = image.clone();
	assert(!newFrame.empty());
	std::vector<std::vector<int>> vecSeams;

	for(int i = 0; i < seams; i++)
	{
		//Gradient Magnitude for intensity of image.
		cv::Mat gradientMagnitude = computeGradientMagnitude(newFrame);
		//Use DP to create the real energy map that is used for path calculation.
		// Strictly using vertical paths for testing simplicity.
		cv::Mat pathIntensityMat = computePathIntensityMat(gradientMagnitude);

		if(pathIntensityMat.rows == 0 && pathIntensityMat.cols == 0)
			return false;
		std::vector<int> seam = getLeastImportantPath(pathIntensityMat);
		vecSeams.push_back(seam);
		if(seamsVect)
			seamsVect->push_back(seam);

		newFrame = removeLeastImportantPath(newFrame, seam);

		if(newFrame.rows == 0 || newFrame.cols == 0)
			return false;
	}

	if (grow)
	{
		cv::Mat growMat = image.clone();

		for(size_t i = 0; i < vecSeams.size(); i++)
		{
			growMat = addLeastImportantPath(growMat,vecSeams[i]);
		}
		image = growMat;
	}
	else
	{
		image = newFrame;
	}
	return true;
}

bool SeamCarving::strechImageVert(cv::Mat& image, int seams, bool grow, std::vector<std::vector<int>>* seamsVect)
{
	cv::transpose(image, image);
	bool ret = strechImage(image, seams, grow, seamsVect);
	cv::transpose(image, image);
	return ret;
}

bool SeamCarving::strechImageWithSeamsImage(cv::Mat& image, cv::Mat& seamsImage, int seams, bool grow)
{
	std::vector<std::vector<int>> seamsVect;
	seamsImage = image.clone();

	bool ret = SeamCarving::strechImage(image, seams, grow, &seamsVect);
	if(!ret)
		return false;

	for(size_t i = 0; i < seamsVect.size(); ++i)
		seamsImage = drawSeam(seamsImage, seamsVect[i]);
	return true;
}

cv::Mat SeamCarving::GetEnergyImg(const cv::Mat &img)
{
	// find partial derivative of x-axis and y-axis seperately
	// sum up the partial derivates
	float pd[] = {1, 2, 1, 0, 0, 0, -1, -2 - 1};
	cv::Mat xFilter(3, 3, CV_32FC1, pd);
	cv::Mat yFilter = xFilter.t();
	cv::Mat grayImg;
	cv::cvtColor(img, grayImg, cv::COLOR_RGBA2GRAY);
	cv::Mat dxImg;
	cv::Mat dyImg;

	cv::filter2D(grayImg, dxImg, 0, xFilter);
	cv::filter2D(grayImg, dyImg, 0, yFilter);
	//cv::Mat zeroMat = cv::Mat::zeros(dxImg.rows, dxImg.cols, dxImg.type());
	//cv::Mat absDxImg;
	//cv::Mat absDyImg;
	//cv::absdiff(dxImg, zeroMat, absDxImg);
	//cv::absdiff(dyImg, zeroMat, absDyImg);
	cv::Mat absDxImg = cv::abs(dxImg);
	cv::Mat absDyImg = cv::abs(dyImg);

	cv::Mat energyImg;
	cv::add(absDxImg, absDyImg, energyImg);
	return energyImg;
}

cv::Mat SeamCarving::computeGradientMagnitude(const cv::Mat &frame)
{
	cv::Mat grayScale;
	cv::cvtColor(frame, grayScale, cv::COLOR_RGBA2GRAY);
	cv::Mat drv = cv::Mat(grayScale.size(), CV_16SC1);
	cv::Mat drv32f = cv::Mat(grayScale.size(), CV_32FC1);
	cv::Mat mag = cv::Mat::zeros(grayScale.size(), CV_32FC1);
	Sobel(grayScale, drv, CV_16SC1, 1, 0);
	drv.convertTo(drv32f, CV_32FC1);
	cv::accumulateSquare(drv32f, mag);
	Sobel(grayScale, drv, CV_16SC1, 0, 1);
	drv.convertTo(drv32f, CV_32FC1);
	cv::accumulateSquare(drv32f, mag);
	cv::sqrt(mag, mag);
	return mag;
}

float SeamCarving::intensity(float currIndex, int start, int end)
{
	if(start < 0 || start >= end)
	{
		return FLT_MAX;
	}
	else
	{
		return currIndex;
	}
}

cv::Mat SeamCarving::computePathIntensityMat(const cv::Mat &rawEnergyMap)
{
	cv::Mat pathIntensityMap = cv::Mat(rawEnergyMap.size(), CV_32FC1);

	if(rawEnergyMap.total() == 0 || pathIntensityMap.total() == 0)
	{
		return cv::Mat();
	}

	//First row of intensity paths is the same as the energy map
	rawEnergyMap.row(0).copyTo(pathIntensityMap.row(0));
	float max = 0;

	//The rest of them use the DP calculation using the minimum of the 3 pixels above them + their own intensity.
	for(int row = 1; row < pathIntensityMap.rows; row++)
	{
		for(int col = 0; col < pathIntensityMap.cols; col++)
		{
			//The initial intensity of the pixel is its raw intensity
			float pixelIntensity = rawEnergyMap.at<float>(row, col);
			//The minimum intensity from the current path of the 3 pixels above it is added to its intensity.
			float p1 = intensity(pathIntensityMap.at<float>(row-1, col-1), col - 1, pathIntensityMap.cols);
			float p2 = intensity(pathIntensityMap.at<float>(row-1, col), col, pathIntensityMap.cols);
			float p3 = intensity(pathIntensityMap.at<float>(row-1, col+1), col + 1, pathIntensityMap.cols);

			float minIntensity = std::min(p1, p2);
			minIntensity = std::min(minIntensity, p3);

			pixelIntensity += minIntensity;

			max = std::max(max, pixelIntensity);
			pathIntensityMap.at<float>(row, col) = pixelIntensity;
		}
	}
	return pathIntensityMap;
}

std::vector<int> SeamCarving::getLeastImportantPath(const cv::Mat &importanceMap)
{
	if(importanceMap.total() == 0)
	{
		return std::vector<int>();
	}

	//Find the beginning of the least important path. Trying an averaging approach because absolute min wasn't very reliable.
	float minImportance = importanceMap.at<float>(importanceMap.rows - 1, 0);
	int minCol = 0;
	for (int col = 1; col < importanceMap.cols; col++)
	{
		float currPixel =importanceMap.at<float>(importanceMap.rows - 1, col);
		if(currPixel < minImportance)
		{
			minCol = col;
			minImportance = currPixel;
		}
	}

	std::vector<int> leastEnergySeam(importanceMap.rows);
	leastEnergySeam[importanceMap.rows-1] = minCol;
	for(int row = importanceMap.rows - 2; row >= 0; row--)
	{
		float p1 = intensity(importanceMap.at<float>(row, minCol-1), minCol - 1, importanceMap.cols);
		float p2 = intensity(importanceMap.at<float>(row, minCol), minCol, importanceMap.cols);
		float p3 = intensity(importanceMap.at<float>(row, minCol+1), minCol + 1, importanceMap.cols);
		//Adjust the min column for path following
		if(p1 < p2 && p1 < p3)
		{
			minCol -= 1;
		}
		else if(p3 < p1 && p3 < p2)
		{
			minCol += 1;
		}
		leastEnergySeam[row] = minCol;
	}

	return leastEnergySeam;
}

cv::Mat SeamCarving::removeLeastImportantPath(const cv::Mat &original, const std::vector<int> &seam)
{
	cv::Size orgSize = original.size();
	// new mat needs to shrink by one collumn
	cv::Size size = cv::Size(orgSize.width-1, orgSize.height);
	cv::Mat newMat = cv::Mat(size, original.type());

	for(size_t row = 0; row < seam.size(); row++)
	{
		removePixel(original, newMat, row, seam[row]);
	}
	return newMat;
}

void SeamCarving::removePixel(const cv::Mat &original, cv::Mat &outputMat, int row, int minCol)
{
	int width = original.cols;
	int channels = original.channels();
	int originRowStart = row * channels * width;
	int newRowStart = row * channels * (width - 1);
	int firstNum = minCol * channels;
	unsigned char *rawOrig = original.data;
	unsigned char *rawOutput = outputMat.data;

	//std::cout << "originRowStart: " << originRowStart << std::endl;
	//std::cout << "newRowStart: " << newRowStart << std::endl;
	//std::cout << "firstNum: " << firstNum << std::endl;
	memcpy(rawOutput + newRowStart, rawOrig + originRowStart, firstNum);

	int originRowMid = originRowStart + (minCol + 1) * channels;
	int newRowMid = newRowStart + minCol * channels;
	int secondNum = (width - 1) * channels - firstNum;

	//std::cout << "originRowMid: " << originRowMid << std::endl;
	//std::cout << "newRowMid: " << newRowMid << std::endl;
	//std::cout << "secondNum: " << secondNum << std::endl;
	memcpy(rawOutput + newRowMid, rawOrig + originRowMid, secondNum);

	int leftPixel = minCol - 1;
	int rightPixel = minCol + 1;

	int byte1 = rawOrig[originRowStart + minCol * channels];
	int byte2 = rawOrig[originRowStart + minCol * channels + 1];
	int byte3 = rawOrig[originRowStart + minCol * channels + 2];

	if (rightPixel < width)
	{
		int byte1R = rawOrig[originRowStart + rightPixel * channels];
		int byte2R = rawOrig[originRowStart + rightPixel * channels + 1];
		int byte3R = rawOrig[originRowStart + rightPixel * channels + 2];
		rawOutput[newRowStart + minCol * channels] = (unsigned char)((byte1 + byte1R) / 2);
		rawOutput[newRowStart + minCol * channels + 1] = (unsigned char)((byte2 + byte2R) / 2);
		rawOutput[newRowStart + minCol * channels + 2] = (unsigned char)((byte3 + byte3R) / 2);
	}

	if(leftPixel >= 0)
	{
		int byte1L = rawOrig[originRowStart + leftPixel*channels];
		int byte2L = rawOrig[originRowStart + leftPixel*channels+1];
		int byte3L = rawOrig[originRowStart + leftPixel*channels+2];
		rawOutput[newRowStart + leftPixel*channels] = (unsigned char) ((byte1 + byte1L)/2);
		rawOutput[newRowStart + leftPixel*channels+1] = (unsigned char) ((byte2 + byte2L)/2);
		rawOutput[newRowStart + leftPixel*channels+2] = (unsigned char) ((byte3 + byte3L)/2);
	}
}

cv::Mat SeamCarving::addLeastImportantPath(const cv::Mat &original, const std::vector<int> &seam)
{
	cv::Size orgSize = original.size();
	// new mat needs to grow by one column
	cv::Size size = cv::Size(orgSize.width+1, orgSize.height);
	cv::Mat newMat = cv::Mat(size, original.type());

	for(size_t row = 0; row < seam.size(); row++)
	{
		//std::cout << "row: " << row << ", col: " << seam[row] << std::endl;
		addPixel(original, newMat, row, seam[row]);
	}
	return newMat;
}

void SeamCarving::addPixel(const cv::Mat &original, cv::Mat &outputMat, int row, int minCol)
{
	int width = original.cols;
	int channels = original.channels();
	int originRowStart = row * channels * width;
	int newRowStart = row * channels * (width + 1);
	int firstNum = (minCol + 1) * channels;

	unsigned char *rawOrig = original.data;
	unsigned char *rawOutput = outputMat.data;

	memcpy(rawOutput + newRowStart, rawOrig + originRowStart, firstNum);

	memcpy(rawOutput + newRowStart + firstNum, rawOrig + originRowStart + firstNum, channels);

	int originRowMid = originRowStart + ((minCol + 1) * channels);
	int newRowMid = newRowStart + ((minCol + 2) * channels);
	int secondNum = (width * channels) - firstNum;

	memcpy(rawOutput + newRowMid, rawOrig + originRowMid, secondNum);

	int leftPixel = minCol - 1;
	int rightPixel = minCol + 1;

	int byte1 = rawOrig[originRowStart + minCol * channels];
	int byte2 = rawOrig[originRowStart + minCol * channels + 1];
	int byte3 = rawOrig[originRowStart + minCol * channels + 2];

	if (rightPixel < width)
	{
		int byte1R = rawOrig[originRowStart + rightPixel * channels];
		int byte2R = rawOrig[originRowStart + rightPixel * channels + 1];
		int byte3R = rawOrig[originRowStart + rightPixel * channels + 2];
		rawOutput[newRowStart + minCol * channels] = (unsigned char)((byte1 + byte1R) / 2);
		rawOutput[newRowStart + minCol * channels + 1] = (unsigned char)((byte2 + byte2R) / 2);
		rawOutput[newRowStart + minCol * channels + 2] = (unsigned char)((byte3 + byte3R) / 2);
	}

	if(leftPixel >= 0)
	{
		int byte1L = rawOrig[originRowStart + leftPixel*channels];
		int byte2L = rawOrig[originRowStart + leftPixel*channels+1];
		int byte3L = rawOrig[originRowStart + leftPixel*channels+2];
		rawOutput[newRowStart + leftPixel*channels] = (unsigned char) ((byte1 + byte1L)/2);
		rawOutput[newRowStart + leftPixel*channels+1] = (unsigned char) ((byte2 + byte2L)/2);
		rawOutput[newRowStart + leftPixel*channels+2] = (unsigned char) ((byte3 + byte3L)/2);
	}
}

cv::Mat SeamCarving::drawSeam(const cv::Mat &frame, const std::vector<int> &seam)
{
	cv::Mat retMat = frame.clone();
	for(int row = 0; row < frame.rows; row++)
	{
		for(int col = 0; col < frame.cols; col++)
		{
			retMat.at<cv::Vec3b>(row, seam[row])[0] = 0;
			retMat.at<cv::Vec3b>(row, seam[row])[1] = 255;
			retMat.at<cv::Vec3b>(row, seam[row])[2] = 0;
		}
	}
	return retMat;
}
