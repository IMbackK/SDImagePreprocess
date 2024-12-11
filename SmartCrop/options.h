/* * SmartCrop - A tool for content aware croping of images
 * Copyright (C) 2024 Carl Philipp Klemm
 *
 * This file is part of SmartCrop.
 *
 * SmartCrop is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SmartCrop is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SmartCrop.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
#include <string>
#include <vector>
#include <argp.h>
#include <iostream>
#include <filesystem>
#include <opencv2/core/types.hpp>
#include "log.h"

const char *argp_program_version = "AIImagePreprocesses";
const char *argp_program_bug_address = "<carl@uvos.xyz>";
static char doc[] = "Application that trainsforms images into formats, sizes and aspect ratios required for ai training";
static char args_doc[] = "FILE(S)";

static struct argp_option options[] =
{
  {"verbose",		'v', 0,				0,	"Show debug messages" },
  {"quiet", 		'q', 0,				0,	"only output data" },
  {"model", 		'm', "[FILENAME]",	0,	"YoloV8 model to use for detection" },
  {"classes", 		'c', "[FILENAME]",	0,	"classes text file to use" },
  {"out",	 		'o', "[DIRECTORY]",	0,	"directory whre images are to be saved" },
  {"debug", 		'd', 0,				0,	"output debug images" },
  {"seam-carving", 	's', 0,				0,	"use seam carving to change image aspect ratio instead of croping"},
  {"x-size", 		'x', "[PIXELS]",	0,	"target output width, default: 1024"},
  {"y-size", 		'y', "[PIXELS]",	0,	"target output height, default: 1024"},
  {"focus-person",	'f', "[FILENAME]",	0,	"a file name to an image of a person that the crop should focus on"},
  {"person-threshold",	't', "[NUMBER]",	0,	"the threshold at witch to consider a person matched, defaults to 0.363"},
  {0}
};

struct Config
{
	std::vector<std::filesystem::path> imagePaths;
	std::filesystem::path modelPath;
	std::filesystem::path classesPath;
	std::filesystem::path outputDir;
	std::filesystem::path focusPersonImage;
	bool seamCarving = false;
	bool debug = false;
	double threshold = 0.363;
	cv::Size targetSize = cv::Size(1024, 1024);
};

static error_t parse_opt (int key, char *arg, struct argp_state *state)
{
	Config *config = reinterpret_cast<Config*>(state->input);
	try
	{
		switch (key)
		{
		case 'q':
			Log::level = Log::ERROR;
			break;
		case 'v':
			Log::level = Log::DEBUG;
			break;
		case 'm':
			config->modelPath = arg;
			break;
		case 'c':
			config->classesPath = arg;
			break;
		case 'd':
			config->debug = true;
			break;
		case 'o':
			config->outputDir.assign(arg);
			break;
		case 's':
			config->seamCarving = true;
			break;
		case 'f':
			config->focusPersonImage = arg;
			break;
		case 't':
			config->threshold = std::atof(arg);
			break;
		case 'x':
		{
			int x = std::stoi(arg);
			config->targetSize = cv::Size(x, config->targetSize.height);
			break;
		}
		case 'y':
		{
			int y = std::stoi(arg);
			config->targetSize = cv::Size(config->targetSize.width, y);
			break;
		}
		case ARGP_KEY_ARG:
			config->imagePaths.push_back(arg);
			break;
		default:
			return ARGP_ERR_UNKNOWN;
		}
	}
	catch(const std::invalid_argument& ex)
	{
		std::cout<<arg<<" passed for argument -"<<static_cast<char>(key)<<" is not a valid number.\n";
		return ARGP_KEY_ERROR;
	}
	return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};
