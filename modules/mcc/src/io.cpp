// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
 * MIT License
 *
 * Copyright (c) 2018 Pedro Diamel Marrero Fernández
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


#include "opencv2/mcc/io.hpp"
namespace cv
{
namespace ccm
{
IO::IO(std::string illuminant_, std::string observer_) :illuminant(illuminant_), observer(observer_) {};

bool IO::operator<(const IO& other) const
{
    return (illuminant < other.illuminant || ((illuminant == other.illuminant) && (observer < other.observer)));
}

bool IO::operator==(const IO& other) const
{
    return illuminant == other.illuminant && observer == other.observer;
};

// data from https://en.wikipedia.org/wiki/Standard_illuminant.
std::vector<double> xyY2XYZ(const std::vector<double>& xyY)
{
    double Y = xyY.size() >= 3 ? xyY[2] : 1;
    return { Y * xyY[0] / xyY[1], Y, Y / xyY[1] * (1 - xyY[0] - xyY[1]) };
}

/* *\ brief function to get illuminants*/
std::map <IO, std::vector<double>> getIlluminant()
{
    std::map <IO, std::vector<double>>  illuminants_;
    for (auto it = illuminants_xy.begin(); it != illuminants_xy.end(); ++it)
    {
        illuminants_[it->first] = xyY2XYZ(it->second);
    }
    return illuminants_;
}

} // namespace ccm
} // namespace cv