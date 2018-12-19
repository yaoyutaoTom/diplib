/*
 * DIPlib 3.0
 * This file contains definitions of various Hough Transform functions.
 *
 * (c)2018, Cris Luengo.
 * Based on original DIPimage code: (c)1999-2014, Delft University of Technology.
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

#include "diplib.h"
#include "diplib/detection.h"
#include "diplib/generation.h"
#include "diplib/transform.h"
#include "diplib/generic_iterators.h"

namespace dip {

namespace {

enum class RadonTransformCirclesMode { full, projection, subpixelProjection };

enum class RadonTransformCirclesOption { normalize, correct, hollow, filled, detectMaxima, saveParamSpace };
DIP_DECLARE_OPTIONS( RadonTransformCirclesOption, RadonTransformCirclesOptions )

void CreateSphere(
      Image& sphere, // Already forged, should be of type SFLOAT or DFLOAT
      dfloat radius,
      dfloat sigma,
      RadonTransformCirclesOptions options
) {
   dip::uint nDims = sphere.Dimensionality();
   if( options.Contains( RadonTransformCirclesOption::correct )) {
      dfloat tmp = 0.25 * radius * radius - static_cast< dfloat >( nDims - 1 ) * sigma * sigma;
      tmp = std::max( tmp, 0.0 ); // prevent square root of negative value
      radius = 0.5 * radius + std::sqrt( tmp );
   }
   dfloat sphereValue = std::sqrt( 2.0 * pi ) * sigma;
   dfloat innerRadius = radius - 3.0 * sigma;
   dfloat innerValue = options.Contains( RadonTransformCirclesOption::hollow ) ? -std::max( 1.0, innerRadius * innerRadius ) : -1.0;
   if( options.Contains( RadonTransformCirclesOption::normalize )) {
      sphereValue /= HypersphereSurface( nDims, radius );
      if( options.Contains( RadonTransformCirclesOption::filled )) {
         innerValue /= std::max( 1.0, HypersphereVolume( nDims, innerRadius )); // TODO!
      } else if( options.Contains( RadonTransformCirclesOption::hollow )) {
         innerValue /= std::max( 1.0, HypersphereSurface( nDims, innerRadius ));
      }
   }
   auto sz = static_cast< dip::uint >( 1 + 2 * ( std::ceil( radius ) + std::ceil( 3 * sigma )));
   UnsignedArray sizes( nDims, sz );
   sphere.Fill( 0 );
   FloatArray origin = sphere.GetCenter();
   DrawBandlimitedBall( sphere, 2 * radius, origin, { sphereValue }, S::EMPTY, sigma );
   if( options.Contains( RadonTransformCirclesOption::filled )) {
      DrawBandlimitedBall( sphere, 2 * innerRadius, origin, { -innerValue }, S::FILLED, sigma );
   } else if( options.Contains( RadonTransformCirclesOption::hollow )) {
      DrawBandlimitedBall( sphere, 2 * innerRadius, origin, { -innerValue }, S::EMPTY, sigma );
   }
}

void ComputeFullParameterSpace(
      Image const& inFT,
      Image& paramSpace,
      Range const& radii,
      dfloat sigma,
      RadonTransformCirclesOptions options
) {
   UnsignedArray outSize = inFT.Sizes();
   Image sphere( outSize, 1, DT_SFLOAT );
   Image sphereFT;
   outSize.push_back( radii.Size() );
   paramSpace.ReForge( outSize, 1, DT_SFLOAT );
   ImageSliceIterator dest( paramSpace, inFT.Dimensionality() );
   for( auto radius = radii.begin(); radius != radii.end(); ++radius, ++dest ) {
      CreateSphere( sphere, *radius, sigma, options );
      FourierTransform( sphere, sphereFT );
      sphereFT *= inFT;
      FourierTransform( sphereFT, *dest, { S::INVERSE, S::REAL } );
   }
}

} // namespace

RadonCircleParametersArray RadonTransformCircles(
      Image const& in,
      Image& out,
      Range radii,
      dfloat sigma,
      String const& s_mode,
      StringSet const& s_options
) {
   DIP_THROW_IF( !in.IsForged(), E::IMAGE_NOT_FORGED );
   DIP_THROW_IF( !in.IsScalar(), E::IMAGE_NOT_SCALAR );
   DIP_THROW_IF( in.Dimensionality() < 1, E::DIMENSIONALITY_NOT_SUPPORTED );
   DIP_THROW_IF( in.DataType().IsComplex(), E::DATA_TYPE_NOT_SUPPORTED );
   DIP_THROW_IF( sigma < 0.8, E::PARAMETER_OUT_OF_RANGE );
   DIP_THROW_IF( radii.start < static_cast< dip::sint >( std::ceil( 3 * sigma )) - 1, E::PARAMETER_OUT_OF_RANGE );
   DIP_THROW_IF( radii.stop > static_cast< dip::sint >( in.Sizes().minimum_value() ), E::PARAMETER_OUT_OF_RANGE );
   DIP_THROW_IF( radii.step < 1, E::PARAMETER_OUT_OF_RANGE );

   // Modes:
   // - "full": Returns the full parameter space, an image of the size of `in` with an additional dimension of the size of `r`.
   // - "projection": Returns two images of the size of `in` (in the form of two tensor elements), containing the max and argmax along the r axis.
   // - "subpixel projection": Idem, but the argmax is computed with sub-pixel precision. It stores 3 slices along r at the time,
   //                          and looks for local maxima along the r axis by fitting a parabola to the the 3 samples.
   RadonTransformCirclesMode mode;
   if( s_mode == S::FULL ) {
      mode = RadonTransformCirclesMode::full;
   } else if( s_mode == S::PROJECTION ) {
      mode = RadonTransformCirclesMode::projection;
   } else if( s_mode == S::SUBPIXEL_PROJECTION ) {
      mode = RadonTransformCirclesMode::subpixelProjection;
   } else {
      DIP_THROW_INVALID_FLAG( s_mode );
   }

   // Options:
   // - "normalize": makes it so that larger circles don't have a larger maximum
   // - "correct": if also normalized, the size of the template is corrected to reduce bias in the radius estimate
   // - "hollow": add a negative ring just inside the positive ring of the template: forces the algorithm to look for rings, not disks
   // - "filled": fill the positive ring with negative values: forces the algorithm to look for rings without anything in them
   // - "no maxima detection": the `RadonCircleParametersArray` output is an empty array.
   // - "no parameter space": the `out` image is not used, the parameter space is potentially computed in chunks to save memory.
   RadonTransformCirclesOptions options = RadonTransformCirclesOption::detectMaxima + RadonTransformCirclesOption::saveParamSpace;
   for( auto& s : s_options ) {
      if( s == S::NORMALIZE ) {
         options += RadonTransformCirclesOption::normalize;
      } else if( s == S::CORRECT ) {
         options += RadonTransformCirclesOption::correct;
      } else if( s == S::HOLLOW ) {
         options += RadonTransformCirclesOption::hollow;
      } else if( s == S::FILLED ) {
         options += RadonTransformCirclesOption::filled;
      } else if( s == S::NO_MAXIMA_DETECTION ) {
         options -= RadonTransformCirclesOption::detectMaxima;
      } else if( s == S::NO_PARAMETER_SPACE ) {
         options -= RadonTransformCirclesOption::saveParamSpace;
      } else {
         DIP_THROW_INVALID_FLAG( s );
      }
   }
   DIP_THROW_IF( !options.Contains( RadonTransformCirclesOption::saveParamSpace ) &&
                 !options.Contains( RadonTransformCirclesOption::detectMaxima ),
                 "Both \"no maxima detection\" and \"no parameter space\" options were given -- nothing to do" );
   if( !options.Contains( RadonTransformCirclesOption::normalize )) {
      options -= RadonTransformCirclesOption::correct; // Never correct if we don't normalize
   }

   // Prepare
   Image inFT = FourierTransform( in ); // TODO: We could try using the "fast" option, leading to a slightly larger parameter space.
   Image tmp_paramSpace;
   Image& parameterSpace = options.Contains( RadonTransformCirclesOption::saveParamSpace ) ? out : tmp_paramSpace;
   RadonCircleParametersArray out_params;

   // Compute parameter space
   switch( mode ) {
      case RadonTransformCirclesMode::full:
         if ( options.Contains( RadonTransformCirclesOption::saveParamSpace )) {
            ComputeFullParameterSpace( inFT, parameterSpace, radii, sigma, options );
         } else {
            // TODO: compute `parameterSpace` in chunks and fill `out_params`
            return out_params;
         }
         break;
      case RadonTransformCirclesMode::projection:
         // TODO
         break;
      case RadonTransformCirclesMode::subpixelProjection:
         // TODO
         break;
   }

   // Find circle parameters
   if( options.Contains( RadonTransformCirclesOption::detectMaxima )) {
      // TODO
   }
   return out_params;
}

} // namespace
