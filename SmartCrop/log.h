/**
* eisgenerator
* Copyright (C) 2021 Carl Klemm
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* version 3 as published by the Free Software Foundation.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the
* Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
* Boston, MA  02110-1301, USA.
*/

#pragma once
#include <iostream>
#include <string>

class Log
{
public:

	enum Level
	{
		DEBUG,
		INFO,
		WARN,
		ERROR
	};

private:
	bool opened = false;
	Level msglevel = DEBUG;
	bool endline = true;

	std::string getLabel(Level level);

public:

	static bool headers;
	static Level level;

	Log() {}
	Log(Level type, bool endlineI = true);
	~Log();

	template<class T> Log &operator<<(const T &msg)
	{
		if(msglevel >= level)
		{
			if(msglevel == ERROR)
				std::cerr<<msg;
			else
				std::cout<<msg;
			opened = true;
		}
		return *this;
	}
};
