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

#include "tokenize.h"


std::vector<std::string> tokenizeBinaryIgnore(const std::string& str, const char delim, const char ignoreBraket, const char escapeChar)
{
	std::vector<std::string> tokens;
	std::string token;
	bool inBaracket = false;
	for(size_t i = 0; i < str.size(); ++i)
	{
		if(str[i] == delim && !inBaracket && (i == 0 || str[i-1] != escapeChar))
		{
			tokens.push_back(token);
			token.clear();
		}
		else
		{
			token.push_back(str[i]);
		}
		if(ignoreBraket == str[i])
			inBaracket = !inBaracket;
	}
	if(!inBaracket)
		tokens.push_back(token);
	return tokens;
}
