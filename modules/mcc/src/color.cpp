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

#include "precomp.hpp"

namespace cv
{
namespace ccm
{

Color Color::to(const ColorSpace& other, CAM method, bool save)
{
    if (history.count(other) == 1)
    {

        return *history[other];
    }
    if (cs.relate(other))
    {
        return Color(cs.relation(other).run(colors), other);
    }
    Operations ops;
    ops.add(cs.to).add(XYZ(cs.io).cam(other.io, method)).add(other.from);
    std::shared_ptr<Color> color(new Color(ops.run(colors), other));
    if (save)
    {
        history[other] = color;
    }
    return *color;
}

cv::Mat Color::channel(cv::Mat m, int i)
{
    cv::Mat dchannels[3];
    split(m, dchannels);
    return dchannels[i];
}

cv::Mat Color::toGray(IO io, CAM method, bool save)
{
    XYZ xyz(io);
    return channel(this->to(xyz, method, save).colors, 1);
}

cv::Mat Color::toLuminant(IO io, CAM method, bool save)
{
    Lab lab(io);
    return channel(this->to(lab, method, save).colors, 0);
}

cv::Mat Color::diff(Color& other, DISTANCE_TYPE method)
{
    return diff(other, cs.io, method);
}

cv::Mat Color::diff(Color& other, IO io, DISTANCE_TYPE method)
{
    Lab lab(io);
    switch (method)
    {
    case cv::ccm::CIE76:
    case cv::ccm::CIE94_GRAPHIC_ARTS:
    case cv::ccm::CIE94_TEXTILES:
    case cv::ccm::CIE2000:
    case cv::ccm::CMC_1TO1:
    case cv::ccm::CMC_2TO1:
        return distance(to(lab).colors, other.to(lab).colors, method);
    case cv::ccm::RGB:
        return distance(to(*cs.nl).colors, other.to(*cs.nl).colors, method);
    case cv::ccm::RGBL:
        return distance(to(*cs.l).colors, other.to(*cs.l).colors, method);
    default:
        throw std::invalid_argument { "Wrong method!" };
        break;
    }
}

void Color::getGray(double JDN)
{
    cv::Mat lab = to(Lab_D65_2).colors;
    cv::Mat gray(colors.size(), colors.type());
    int fromto[] = { 0,0, -1,1, -1,2 };
    mixChannels(&lab, 1, &gray, 1, fromto, 3);
    cv::Mat d = distance(lab, gray, CIE2000);
    this->grays = d < JDN;
    this->colored = ~grays;
}

Color Color::operator[](cv::Mat mask)
{
    return Color(maskCopyTo(colors, mask), cs);
}

Color Color::operator=(Color inp)
{
    return inp;
}

} // namespace ccm
} // namespace cv