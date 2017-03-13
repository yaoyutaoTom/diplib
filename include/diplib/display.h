/*
 * DIPlib 3.0
 * This file contains declarations for the ImageDisplay function
 *
 * (c)2017, Cris Luengo.
 * Based on original DIPlib code: (c)1995-2014, Delft University of Technology.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DIP_DISPLAY_H
#define DIP_DISPLAY_H

#include "diplib.h"


namespace dip {


/// \brief Parameters to the `dip::ImageDisplay` function.
struct ImageDisplayParams {
   String mode; ///< "lin" (for linear), "log" (for logarithmic), "based" (for based at 0, where 0 is anchored at grey value 128.
   String complex; ///< "mag" (for magnitude) or "abs", "phase", "real", "imag".
   String projection; ///< "slice", "max", "mean".
   dfloat lowerBound; ///< grey value to set to 0.
   dfloat upperBound; ///< grey value to set to 255.
};

/// \brief Transform the image to make it suitable for display.
///
/// The image `in` will be transformed from nD to 2D according to `params.projection`, such that original dimension
/// `dim1` becomes the first dimension and `dim2` becomes the second. In the case of "slice" projection, `coordinates`
/// indicates which slice to extract; it gives the coordinates to a pixel that will be visible in the output.
/// `in` must have at least two dimensions. If `in` has exactly two dimensions, `coordinates`, `dim1` and `dim2` are
/// ignored.
///
/// `params.lowerBound` and `params.upperBound` indicate how the grey-values will be stretched to the range of the
/// `dip::DT_UINT8` output data type. In case `in` is complex, it will be converted to real values through
/// `params.complex`. Finally, if `params.mode` is "log", a logarithmic stretching will be applied, instead of linear.
/// With the "based" mode, the lower and upper bounds will be adjusted such that 0 is mapped to middle grey. Note that
/// the upper bound must be stricly greater than the lower bound.
///
/// If `in` is a color image, it will be converted to RGB, and each of the three channels will be handled identically.
/// Other tensor images are not supported.
void ImageDisplay(
      Image const& in,
      Image& out,
      UnsignedArray const& coordinates,
      dip::uint dim1,
      dip::uint dim2,
      ImageDisplayParams const& params
);

inline Image ImageDisplay (
      Image const& in,
      UnsignedArray const& coordinates,
      dip::uint dim1,
      dip::uint dim2,
      ImageDisplayParams const& params
) {
   Image out;
   ImageDisplay( in, out, coordinates, dim1, dim2, params );
   return out;
}


} // namespace dip

#endif // DIP_DISPLAY_H