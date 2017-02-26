/*
 * DIPlib 3.0
 * This file contains definitions for the Image class and related functions.
 *
 * (c)2014-2017, Cris Luengo.
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

#include <cstdlib>   // std::malloc, std::realloc, std::free
#include <limits>
#include <algorithm>

#include "diplib.h"


namespace dip {


//
// --- Internal functions, static ---
//


// Compute a normal stride array.
static void ComputeStrides(
      UnsignedArray const& sizes,
      dip::uint s,               // set to tensor.Elements()
      IntegerArray& strides
) {
   dip::uint n = sizes.size();
   strides.resize( n );
   for( dip::uint ii = 0; ii < n; ++ii ) {
      strides[ ii ] = s;
      s *= sizes[ ii ];
   }
}


// Return the number of pixels defined by the sizes array.
// Same as dip::Image::NumberOfPixels() but with check.
static dip::uint FindNumberOfPixels(
      UnsignedArray const& sizes
) {
   dip::uint n = 1;
   for( dip::uint ii = 0; ii < sizes.size(); ++ii ) {
      DIP_THROW_IF( ( sizes[ ii ] != 0 ) && ( n > std::numeric_limits< dip::uint >::max() / sizes[ ii ] ),
                   E::SIZE_EXCEEDS_LIMIT );
      n *= sizes[ ii ];
   }
   return n;
}


// Return the size of the data block needed to store an image given by
// strides and sizes, as well as the (negative) offset of the block if any
// of the strides are negative.
static void FindDataBlockSizeAndStart(
      IntegerArray const& strides,
      UnsignedArray const& sizes,
      dip::uint& size,
      dip::sint& start
) {
   dip::sint min = 0, max = 0;
   for( dip::uint ii = 0; ii < sizes.size(); ++ii ) {
      dip::sint p = ( sizes[ ii ] - 1 ) * strides[ ii ];
      if( p < 0 ) {
         min += p;
      } else {
         max += p;
      }
   }
   start = min;
   size = static_cast< dip::uint >( max - min + 1 );
}


// Return the simple stride (if it exists), the start pixel (offset into the
// data block), and number of pixels defined by strides and sizes.
// If there is no simple stride, sets sstride==0, and returns false.
// Note that sstride==0 does not indicate an error condition, it is possible
// that the image was singleton-expanded from a 0D image.
static bool FindSimpleStrideSizeAndStart(
      IntegerArray const& strides,
      UnsignedArray const& sizes,
      dip::uint& sstride,
      dip::uint& size,
      dip::sint& start
) {
   if( strides.size() == 0 ) {
      // Special case
      sstride = 1;
      size = 1;
      start = 0;
   } else {
      // Find the simple stride
      sstride = std::numeric_limits< dip::uint >::max();
      for( dip::uint ii = 0; ii < strides.size(); ++ii ) {
         if( sizes[ ii ] > 1 ) {
            sstride = std::min( sstride, static_cast< dip::uint >( std::abs( strides[ ii ] ) ) );
         }
      }
      FindDataBlockSizeAndStart( strides, sizes, size, start );
      if( size != ( FindNumberOfPixels( sizes ) - 1 ) * sstride + 1 ) {
         sstride = 0;
         return false;
      }
   }
   return true;
}


// Compute coordinates of a pixel from an offset.
// Strides array must be all positive, and sorted in increasing order.
static UnsignedArray OffsetToCoordinates(
      dip::uint offset,
      IntegerArray const& strides
) {
   UnsignedArray coord( strides.size() );
   for( dip::uint ii = strides.size(); ii > 0; ) {
      // This loop increases its counter at the start, a disadvantage of using
      // unsigned integers as loop counters.
      --ii;
      coord[ ii ] = offset / strides[ ii ];
      offset = offset % strides[ ii ];
   }
   return coord;
}


// Removes elements from the stride array that correspond to singleton dimensions (sizes[ii]==1).
// Modifies the strides array.
void RemoveSingletonsFromStrideArray( UnsignedArray const& sizes, IntegerArray strides ) {
   dip::uint jj = 0;
   for( dip::uint ii = 0; ii < strides.size(); ++ii ) {
      if( sizes[ ii ] > 1 ) {
         strides[ jj ] = strides[ ii ];
         ++jj;
      }
   }
   strides.resize( jj );
}

//
// --- Library functions ---
//


// Constructor.
CoordinatesComputer::CoordinatesComputer( UnsignedArray const& sizes, IntegerArray const& strides ) {
   dip::uint N = strides.size();
   DIP_THROW_IF( sizes.size() != N, "Input arrays do not have the same size" );
   strides_ = strides;
   sizes_.resize( N );
   index_.resize( N );
   offset_ = 0;
   // Set indices to all non-singleton dimensions.
   // Zero-stride dimensions are those that used to be singleton, but were expanded
   // by setting the size > 1 and stride = 0.
   dip::uint nelem = 0;
   for( dip::uint ii = 0; ii < N; ++ii ) {
      sizes_[ ii ] = sizes[ ii ];
      if( ( sizes_[ ii ] != 1 ) && ( strides_[ ii ] != 0 ) ) {
         index_[ nelem ] = ii;
         ++nelem;
         if( strides_[ ii ] < 0 ) {
            // For negative strides, we save a positive value instead. The sizes
            // we make negative to remember that the stride was negative. We also
            // compute the offset that will be needed to compute coordinates.
            // In short, what this accomplishes is to reverse the dimension,
            // making the stride positive but counting starting at the end of
            // the image line instead of at the beginning. The computed coordinate
            // will thus need to be reversed again.
            strides_[ ii ] = -strides_[ ii ];
            offset_ += strides_[ ii ] * ( sizes_[ ii ] - 1 );
            sizes_[ ii ] = -sizes_[ ii ];
         }
      }
   }
   // Sort the indices large to small.
   for( dip::uint ii = 1; ii < nelem; ++ii ) {
      dip::uint keepIndex = index_[ ii ];
      dip::sint key = strides_[ keepIndex ];
      dip::uint jj = ii;
      while( ( jj > 0 ) && ( strides_[ index_[ jj - 1 ] ] < key ) ) {
         index_[ jj ] = index_[ jj - 1 ];
         --jj;
      }
      index_[ jj ] = keepIndex;
   }
   // The indices for the singleton dimensions go at the end.
   for( dip::uint ii = 0; ii < N; ++ii ) {
      if( ( sizes_[ ii ] == 1 ) || ( strides_[ ii ] == 0 ) ) {
         index_[ nelem ] = ii;
         ++nelem;
         // By the time we use these elements, the resudue should be 0, so it
         // doesn't matter what the stride values are. As long as they are not 0!
         strides_[ ii ] = 1;
      }
   }
   //std::cout << "   CoordinatesComputer: \n";
   //std::cout << "      strides_ = " << strides_ << std::endl
   //std::cout << "      sizes_ = " << sizes_ << std::endl
   //std::cout << "      index_ = " << index_ << std::endl;
   //std::cout << "      offset_ = " << offset_ << std::endl;
}

// The function that computes OffsetToCoordinates and IndexToCoordinates.
UnsignedArray CoordinatesComputer::operator()( dip::sint offset ) const {
   dip::uint N = strides_.size();
   UnsignedArray coordinates( N );
   offset += offset_;
   for( dip::uint ii = 0; ii < N; ++ii ) {
      dip::uint jj = index_[ ii ];
      coordinates[ jj ] = static_cast< dip::uint >( offset / strides_[ jj ] );
      offset = offset % strides_[ jj ];
      if( sizes_[ jj ] < 0 ) {
         // This dimension had a negative stride. The computed coordinate started
         // at the end of the line instead of the begging, so we reverse it.
         coordinates[ jj ] = -sizes_[ jj ] - coordinates[ jj ] - 1;
      }
   }
   return coordinates;
}


// Normal strides are the default ones:
// increasing in value, and with contiguous data.
bool Image::HasNormalStrides() const {
   DIP_THROW_IF( !IsForged(), E::IMAGE_NOT_FORGED );
   if( tensorStride_ != 1 ) {
      return false;
   }
   dip::sint total = tensor_.Elements();
   for( dip::uint ii = 0; ii < sizes_.size(); ++ii ) {
      if( strides_[ ii ] != total ) {
         return false;
      }
      total *= sizes_[ ii ];
   }
   return true;
}


// Return a pointer to the start of the data and a single stride to
// walk through all pixels. If this is not possible, porigin==nullptr.
void Image::GetSimpleStrideAndOrigin( dip::uint& sstride, void*& porigin ) const {
   DIP_THROW_IF( !IsForged(), E::IMAGE_NOT_FORGED );
   dip::sint start;
   dip::uint size;
   if( FindSimpleStrideSizeAndStart( strides_, sizes_, sstride, size, start ) ) {
      porigin = Pointer( start );
   } else {
      porigin = nullptr;
   }
}


// Are the dimensions ordered in the same way?
bool Image::HasSameDimensionOrder( Image const& other ) const {
   DIP_THROW_IF( !IsForged(), E::IMAGE_NOT_FORGED );
   DIP_THROW_IF( !other.IsForged(), E::IMAGE_NOT_FORGED );
   // Remove singleton dimensions in stride array
   IntegerArray s1 = strides_;
   RemoveSingletonsFromStrideArray( sizes_, s1 );
   // Remove singleton dimensions in other's stride array
   IntegerArray s2 = other.strides_;
   RemoveSingletonsFromStrideArray( other.sizes_, s2 );
   // Compare number of non-singleton dimensions
   if( s1.size() != s2.size() ) {
      return false;
   }
   // We sort s1, keeping s2 in sync. s2 must be sorted also.
   s1.sort( s2 );
   for( dip::uint ii = 1; ii < s2.size(); ++ii ) {
      if( s2[ ii ] < s2[ ii - 1 ] ) {
         return false;
      }
   }
   return true;
}


//
bool Image::HasValidStrides() const {
   // We require that |strides[ii+1]| > |strides[ii]|*(sizes[ii]-1) (after sorting the absolute strides on size)
   if( sizes_.size() != strides_.size() ) {
      return false;
   }
   // Add tensor dimension and stride to the lists
   IntegerArray s = strides_;
   UnsignedArray d = sizes_;
   if( tensor_.Elements() > 1 ) {
      s.push_back( tensorStride_ );
      d.push_back( tensor_.Elements() );
   }
   dip::uint n = s.size();
   if( n < 2 ) {
      // It's a 0D or 1D image, strides are always OK
      return true;
   }
   // Make sure all strides are positive
   for( dip::uint ii = 0; ii < n; ++ii ) {
      s[ ii ] = std::abs( s[ ii ] );
   }
   s.sort( d );
   // Test invariant
   for( dip::uint ii = 0; ii < n - 1; ++ii ) {
      if( s[ ii + 1 ] <= s[ ii ] * ( static_cast< dip::sint >( d[ ii ] ) - 1 )) {
         return false;
      }
   }
   // It's OK
   return true;
}


//
void Image::SetNormalStrides() {
   DIP_THROW_IF( IsForged(), E::IMAGE_NOT_RAW );
   tensorStride_ = 1;                       // We set tensor strides to 1 by default.
   ComputeStrides( sizes_, tensor_.Elements(), strides_ );
}


//
void Image::GetDataBlockSizeAndStart( dip::uint& size, dip::sint& start ) const {
   FindDataBlockSizeAndStart( strides_, sizes_, size, start );
}
void Image::GetDataBlockSizeAndStartWithTensor( dip::uint& size, dip::sint& start ) const {
   if( tensor_.Elements() > 1 ) {
      UnsignedArray d = sizes_;
      d.push_back( tensor_.Elements() );
      IntegerArray s = strides_;
      s.push_back( tensorStride_ );
      FindDataBlockSizeAndStart( s, d, size, start );
   } else {
      FindDataBlockSizeAndStart( strides_, sizes_, size, start );
   }
}


// Does writing in this image change the data of the other image?
bool Image::Aliases( Image const& other ) const {
   DIP_THROW_IF( !IsForged(), E::IMAGE_NOT_FORGED );
   DIP_THROW_IF( !other.IsForged(), E::IMAGE_NOT_FORGED );

   // Different data blocks do not overlap by definition
   if( dataBlock_ != other.dataBlock_ ) {
      return false;
   }

   // Quicky: if the origin is the same, they share at least one pixel
   dip::uint origin1 = static_cast< uint8* >( origin_ ) - static_cast< uint8* >( dataBlock_.get() );
   dip::uint origin2 = static_cast< uint8* >( other.origin_ ) - static_cast< uint8* >( other.dataBlock_.get() );
   if( origin1 == origin2 ) {
      return true;
   }

   // Copy size and stride arrays, add tensor dimension
   IntegerArray strides1 = strides_;
   UnsignedArray sizes1 = sizes_;
   if( tensor_.Elements() > 1 ) {
      strides1.push_back( tensorStride_ );
      sizes1.push_back( tensor_.Elements() );
   }
   dip::uint ndims1 = strides1.size();
   IntegerArray strides2 = other.strides_;
   UnsignedArray sizes2 = other.sizes_;
   if( other.tensor_.Elements() > 1 ) {
      strides2.push_back( other.tensorStride_ );
      sizes2.push_back( other.tensor_.Elements() );
   }
   dip::uint ndims2 = strides2.size();

   // Check sample sizes
   dip::uint dts1 = dataType_.SizeOf();
   dip::uint dts2 = other.dataType_.SizeOf();
   dip::uint dts = dts1;
   if( dts1 > dts2 ) {
      // Split the samples of 1, adding a new dimension
      dts = dts2;
      dip::uint n = dts1 / dts; // this is always an integer value, samples have size 1, 2, 4, 8 or 16.
      for( dip::uint ii = 0; ii < ndims1; ++ii ) {
         strides1[ ii ] *= n;
      }
      strides1.push_back( 1 );
      sizes1.push_back( n );
      ++ndims1;
   } else if( dts1 < dts2 ) {
      // Split the samples of 2, adding a new dimension
      dip::uint n = dts2 / dts; // this is always an integer value, samples have size 1, 2, 4, 8 or 16.
      for( dip::uint ii = 0; ii < ndims2; ++ii ) {
         strides2[ ii ] *= n;
      }
      strides2.push_back( 1 );
      sizes2.push_back( n );
      ++ndims2;
   } // else, the samples have the same size

   // Make origin in units of data size
   origin1 /= dts;
   origin2 /= dts;

   // Quicky: if both have simple strides larger than one, and their offsets
   // do not differ by a multiple of that stride, they don't overlap.
   dip::uint sstride1, sstride2;
   dip::uint size1, size2;
   dip::sint start1, start2;
   FindSimpleStrideSizeAndStart( strides1, sizes1, sstride1, size1, start1 );
   FindSimpleStrideSizeAndStart( strides2, sizes2, sstride2, size2, start2 );
   start1 += origin1;
   start2 += origin2;
   if( ( sstride1 > 1 ) && ( sstride1 == sstride2 ) ) {
      if( ( start1 - start2 ) % sstride1 != 0 ) {
         return false;
      }
   }

   // Non-overlapping portions of the data block
   if(( start1 + static_cast< dip::sint >( size1 ) <= start2 ) || ( start2 + static_cast< dip::sint >( size2 ) <= start1 )) {
      return false;
   }

   // Lastly, check sizes and strides
   // This is a bit complex

   // Remove singleton dimensions
   for( dip::uint ii = 0; ii < ndims1; ++ii ) {
      if( sizes1[ ii ] == 1 ) {
         sizes1.erase( ii );
         strides1.erase( ii );
         --ii;
         --ndims1;
      }
   }
   for( dip::uint ii = 0; ii < ndims2; ++ii ) {
      if( sizes2[ ii ] == 1 ) {
         sizes2.erase( ii );
         strides2.erase( ii );
         --ii;
         --ndims2;
      }
   }

   // Make sure all strides are positive (un-mirror)
   for( dip::uint ii = 0; ii < ndims1; ++ii ) {
      if( strides1[ ii ] < 0 ) {
         strides1[ ii ] = -strides1[ ii ];
         origin1 -= ( sizes1[ ii ] - 1 ) * strides1[ ii ];
      }
   }
   for( dip::uint ii = 0; ii < ndims2; ++ii ) {
      if( strides2[ ii ] < 0 ) {
         strides2[ ii ] = -strides2[ ii ];
         origin2 -= ( sizes2[ ii ] - 1 ) * strides2[ ii ];
      }
   }

   // Sort strides smallest to largest, keeping sizes in sync.
   strides1.sort( sizes1 );
   strides2.sort( sizes2 );

   // Walk through both stride arrays matching up dimensions
   // The assumed invariant is that stride[ii+1]>=stride[ii]*sizes[ii]

   IntegerArray comstrides;  // common strides
   IntegerArray newstrides1; // new strides img 1
   IntegerArray newstrides2; // new strides img 2
   UnsignedArray newsizes1;    // new sizes img 1
   UnsignedArray newsizes2;    // new sizes img 2

   dip::uint i1 = 0;
   dip::uint i2 = 0;
   while( ( i1 < ndims1 ) && ( strides1[ i1 ] == 0 ) ) { ++i1; }
   while( ( i2 < ndims2 ) && ( strides2[ i2 ] == 0 ) ) { ++i2; }
   while( i1 < ndims1 || i2 < ndims2 ) {
      dip::uint s1 = 0, s2 = 0, d1 = 1, d2 = 1;
      if( i1 < ndims1 ) {
         s1 = static_cast< dip::uint >( strides1[ i1 ] );
         d1 = sizes1[ i1 ];
      }
      if( i2 < ndims2 ) {
         s2 = static_cast< dip::uint >( strides2[ i2 ] );
         d2 = sizes2[ i2 ];
      }
      if( s1 == 0 ) {
         // we're at the end of sizes1
         s1 = s2;
         ++i2;
      } else if( s2 == 0 ) {           // s1 and s2 cannot be 0 at the same time
         // we're at the end of sizes2
         s2 = s1;
         ++i1;
      } else if(( i1 + 1 < ndims1 ) && ( strides1[ i1 + 1 ] <= static_cast< dip::sint >( s2 * ( d2 - 1 )))) {
         // s2 is too large, assume img2 has sizes==1 in this dimension
         s2 = s1;
         d2 = 1;
         ++i1;
      } else if(( i2 + 1 < ndims2 ) && ( strides2[ i2 + 1 ] <= static_cast< dip::sint >( s1 * ( d1 - 1 )))) {
         // s1 is too large, assume img1 has sizes==1 in this dimension
         s1 = s2;
         d1 = 1;
         ++i2;
      } else {
         // matching dimensions
         ++i1;
         ++i2;
      }
      dip::sint cs = comstrides.empty() ? 1 : gcd( s1, s2 ); // The first dimension should have stride==1
      comstrides.push_back( cs );
      newstrides1.push_back( s1 / cs );
      newstrides2.push_back( s2 / cs );
      newsizes1.push_back( d1 );
      newsizes2.push_back( d2 );
   }

   // Compute coordinates of origin for both images
   UnsignedArray neworigin1 = dip::OffsetToCoordinates( origin1, comstrides );
   UnsignedArray neworigin2 = dip::OffsetToCoordinates( origin2, comstrides );

   // Compute, for each of the dimensions, if the views overlap. If
   // they don't overlap for any one dimension, there is no aliasing.
   for( dip::uint ii = 0; ii < comstrides.size(); ++ii ) {
      if( neworigin1[ ii ] + ( newsizes1[ ii ] - 1 ) * newstrides1[ ii ] < neworigin2[ ii ] ) {
         return false;
      }
      if( neworigin2[ ii ] + ( newsizes2[ ii ] - 1 ) * newstrides2[ ii ] < neworigin1[ ii ] ) {
         return false;
      }
      if( ( newstrides1[ ii ] == newstrides2[ ii ] ) &&
          ( newstrides1[ ii ] > 1 ) &&
          ( ( static_cast< dip::sint >( neworigin1[ ii ] ) - static_cast< dip::sint >( neworigin2[ ii ] )) % newstrides1[ ii ] != 0 ) ) {
         return false;
      }
   }

   return true;
}


//
void Image::Forge() {
   if( !IsForged() ) {
      dip::uint size = FindNumberOfPixels( sizes_ );
      DIP_THROW_IF( size == 0, "Cannot forge an image without pixels (sizes must be > 0)" );
      DIP_THROW_IF( TensorElements() > std::numeric_limits< dip::uint >::max() / size,
                   E::SIZE_EXCEEDS_LIMIT );
      size *= TensorElements();
      if( externalInterface_ ) {
         dataBlock_ = externalInterface_->AllocateData( sizes_, strides_, tensor_, tensorStride_, dataType_ );
         // AllocateData() can fail by returning a nullptr. If so, we allocate data in the normal way because we remain raw.
         if( dataBlock_ ) {
            dip::uint sz;
            dip::sint start;
            GetDataBlockSizeAndStartWithTensor( sz, start );
            origin_ = static_cast< uint8* >( dataBlock_.get() ) + start * dataType_.SizeOf();
            //std::cout << "   Successfully forged image with external interface\n";
         }
      }
      if( !IsForged() ) {
         dip::sint start = 0;
         if( HasValidStrides() ) {
            dip::uint sz;
            GetDataBlockSizeAndStartWithTensor( sz, start );
            if( sz != size ) {
               SetNormalStrides();
            }
         } else {
            SetNormalStrides();
         }
         dip::uint sz = dataType_.SizeOf();
         void* p = std::malloc( size * sz );
         DIP_THROW_IF( !p, "Failed to allocate memory" );
         dataBlock_ = std::shared_ptr< void >( p, std::free );
         origin_ = static_cast< uint8* >( p ) + start * sz;
         //std::cout << "   Successfully forged image\n";
      }
   }
}


//
void Image::ReForge(
      UnsignedArray const& sizes,
      dip::uint tensorElems,
      dip::DataType dt,
      Option::AcceptDataTypeChange acceptDataTypeChange
) {
   if(( acceptDataTypeChange == dip::Option::AcceptDataTypeChange::DO_ALLOW ) && protect_ ) {
      dt = dataType_;
   }
   if( IsForged() ) {
      if(( sizes_ == sizes ) && ( tensor_.Elements() == tensorElems ) && ( dataType_ == dt )) {
         // It already matches, nothing to do
         return;
      }
      if(   !protect_ &&
            !IsShared() &&
            HasContiguousData() &&
            ( sizes_.product() * tensor_.Elements() * dataType_.SizeOf() == sizes.product() * tensorElems * dt.SizeOf() )) {
         // The data segment has the right number of bytes, and is not shared with another image: let's reuse it
         dataType_ = dt;
         sizes_ = sizes;
         tensor_.SetVector( tensorElems );
         tensorStride_ = 1;                       // We set tensor strides to 1 by default.
         ComputeStrides( sizes_, tensor_.Elements(), strides_ );
         origin_ = dataBlock_.get();
         return;
      }
   }
   Strip();
   ResetNonDataProperties();
   dataType_ = dt;
   sizes_ = sizes;
   tensor_.SetVector( tensorElems );
   Forge();
}


//
dip::sint Image::Offset( UnsignedArray const& coords ) const {
   DIP_THROW_IF( !IsForged(), E::IMAGE_NOT_FORGED );
   DIP_THROW_IF( coords.size() != sizes_.size(), E::ARRAY_ILLEGAL_SIZE );
   dip::sint offset = 0;
   for( dip::uint ii = 0; ii < sizes_.size(); ++ii ) {
      DIP_THROW_IF( coords[ ii ] >= sizes_[ ii ], E::INDEX_OUT_OF_RANGE );
      offset += coords[ ii ] * strides_[ ii ];
   }
   return offset;
}

//
dip::sint Image::Offset( IntegerArray const& coords ) const {
   DIP_THROW_IF( !IsForged(), E::IMAGE_NOT_FORGED );
   DIP_THROW_IF( coords.size() != sizes_.size(), E::ARRAY_ILLEGAL_SIZE );
   dip::sint offset = 0;
   for( dip::uint ii = 0; ii < sizes_.size(); ++ii ) {
      offset += coords[ ii ] * strides_[ ii ];
   }
   return offset;
}

//
UnsignedArray Image::OffsetToCoordinates( dip::sint offset ) const {
   CoordinatesComputer comp = OffsetToCoordinatesComputer();
   return comp( offset );
}

//
CoordinatesComputer Image::OffsetToCoordinatesComputer() const {
   DIP_THROW_IF( !IsForged(), E::IMAGE_NOT_FORGED );
   return CoordinatesComputer( sizes_, strides_ );
}

//
dip::uint Image::Index( UnsignedArray const& coords ) const {
   DIP_THROW_IF( !IsForged(), E::IMAGE_NOT_FORGED );
   DIP_THROW_IF( coords.size() != sizes_.size(), E::ARRAY_ILLEGAL_SIZE );
   dip::uint index = 0;
   for( dip::uint ii = sizes_.size(); ii > 0; ) {
      --ii;
      DIP_THROW_IF( coords[ ii ] >= sizes_[ ii ], E::INDEX_OUT_OF_RANGE );
      index *= sizes_[ ii ];
      index += coords[ ii ];
   }
   return index;
}

//
UnsignedArray Image::IndexToCoordinates( dip::uint index ) const {
   CoordinatesComputer comp = IndexToCoordinatesComputer();
   return comp( index );
}

//
CoordinatesComputer Image::IndexToCoordinatesComputer() const {
   DIP_THROW_IF( !IsForged(), E::IMAGE_NOT_FORGED );
   IntegerArray fake_strides;
   ComputeStrides( sizes_, 1, fake_strides );
   return CoordinatesComputer( sizes_, fake_strides );
}


} // namespace dip


#ifdef DIP__ENABLE_DOCTEST
#include "doctest.h"
#include <random> // used in one of the tests

DOCTEST_TEST_CASE("[DIPlib] testing the Alias function") {
   dip::Image img1{ dip::UnsignedArray{ 50, 80, 30 }, 3 };
   DOCTEST_REQUIRE( img1.Size( 0 ) == 50 );
   DOCTEST_REQUIRE( img1.Size( 1 ) == 80 );
   DOCTEST_REQUIRE( img1.Size( 2 ) == 30 );
   DOCTEST_REQUIRE( img1.NumberOfPixels() == 120000 );
   DOCTEST_REQUIRE( img1.TensorElements() == 3 );

   dip::Image img2 = img1[ 0 ];
   DOCTEST_CHECK( Alias( img1, img2 ) );
   dip::Image img3 = img1[ 1 ];
   DOCTEST_CHECK( Alias( img1, img3 ) );
   DOCTEST_CHECK_FALSE( Alias( img2, img3 ) );
   dip::Image img4 = img1.At( dip::Range{}, dip::Range{}, dip::Range{ 10 } );
   DOCTEST_CHECK( Alias( img1, img4 ) );
   dip::Image img5 = img1.At( dip::Range{}, dip::Range{}, dip::Range{ 11 } );
   DOCTEST_CHECK_FALSE( Alias( img4, img5 ) );
   dip::Image img6 = img1.At( dip::Range{ 0, -1, 2 }, dip::Range{}, dip::Range{} );
   dip::Image img7 = img1.At( dip::Range{ 1, -1, 2 }, dip::Range{}, dip::Range{} );
   DOCTEST_CHECK( Alias( img1, img7 ) );
   DOCTEST_CHECK_FALSE( Alias( img6, img7 ) );
   img7.Mirror( { true, false, false } );
   DOCTEST_CHECK_FALSE( Alias( img6, img7 ) );
   img7.SwapDimensions( 0, 1 );
   DOCTEST_CHECK_FALSE( Alias( img6, img7 ) );
   dip::Image img8{ dip::UnsignedArray{ 50, 80, 30 }, 3 };
   DOCTEST_CHECK_FALSE( Alias( img1, img8 ) );
   img1.Strip();
   img1.SetDataType( dip::DT_SCOMPLEX );
   img1.Forge();
   DOCTEST_CHECK( Alias( img1, img1.Imaginary() ) );
   DOCTEST_CHECK_FALSE( Alias( img1.Real(), img1.Imaginary() ) );
}

DOCTEST_TEST_CASE("[DIPlib] testing the index and offset computations") {
   std::default_random_engine random;
   std::uniform_int_distribution< dip::uint > randD( 1, 8 );
   std::uniform_int_distribution< dip::uint > randSz( 1, 30 );
   std::uniform_int_distribution< dip::uint > randSSz( 1, 10 ); // Smaller size for 7 or 8 dimensions
   std::uniform_real_distribution< double > randF( 0, 1 );
   for( dip::uint repeat = 0; repeat < 1000; ++repeat ) {
      dip::uint ndims = randD( random );
      dip::UnsignedArray sz( ndims );
      for( dip::uint ii = 0; ii < ndims; ++ii ) {
         sz[ ii ] = ndims > 6 ? randSSz( random ) : randSz( random );
      }
      dip::Image img;
      img.SetSizes( sz );
      try{
         img.Forge();
      } catch (dip::Error &e) {
         std::cout << "Couldn't forge an image of size " << sz << std::endl;
         continue;
      }
      std::uniform_int_distribution< dip::uint > randD( 0, ndims - 1 );
      for( dip::uint ii = 0; ii < randD( random ); ++ii ) {
         img.SwapDimensions( randD( random ), randD( random ) );
      }
      dip::BooleanArray mirror( ndims );
      for( dip::uint ii = 0; ii < ndims; ++ii ) {
         mirror[ ii ] = randF( random ) > 0.7;
      }
      img.Mirror( mirror );
      const dip::UnsignedArray& dims = img.Sizes();
      auto o2c = img.OffsetToCoordinatesComputer();
      auto i2c = img.IndexToCoordinatesComputer();
      for( dip::uint repeat2 = 0; repeat2 < 100; ++repeat2 ) {
         dip::UnsignedArray coords( ndims );
         for( dip::uint ii = 0; ii < ndims; ++ii ) {
            coords[ ii ] = ( dip::uint )std::floor( randF( random ) * dims[ ii ] );
         }
         dip::sint offset = img.Offset( coords );
         DOCTEST_CHECK( o2c( offset ) == coords );
         dip::uint index = img.Index( coords );
         DOCTEST_CHECK( i2c( index ) == coords );
      }
   }
}

#endif // DIP__ENABLE_DOCTEST
