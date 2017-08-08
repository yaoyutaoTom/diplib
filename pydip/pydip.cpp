/*
 * PyDIP 3.0, Python bindings for DIPlib 3.0
 *
 * (c)2017, Flagship Biosciences, Inc., written by Cris Luengo.
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

#include "pydip.h"
#include "diplib/neighborlist.h"

static_assert( sizeof( bool ) == sizeof( dip::bin ), "bool is not one byte, how can I work with logical Python buffers?" );

PYBIND11_PLUGIN( PyDIP_bin ) {
   py::module m( "PyDIP_bin", "DIPlib bindings" );

   auto dtype = py::class_< dip::DataType >( m, "DataType" );
   dtype.def( py::init<>() );
   dtype.def( py::self == py::self );
   dtype.def( "SizeOf", &dip::DataType::SizeOf );
   dtype.def( "IsBinary", &dip::DataType::IsBinary );
   dtype.def( "IsUInt", &dip::DataType::IsUInt );
   dtype.def( "IsSInt", &dip::DataType::IsSInt );
   dtype.def( "IsInteger", &dip::DataType::IsInteger );
   dtype.def( "IsFloat", &dip::DataType::IsFloat );
   dtype.def( "IsReal", &dip::DataType::IsReal );
   dtype.def( "IsComplex", &dip::DataType::IsComplex );
   dtype.def( "IsUnsigned", &dip::DataType::IsUnsigned );
   dtype.def( "IsSigned", &dip::DataType::IsSigned );
   dtype.def( "Real", &dip::DataType::Real );
   dtype.def( "__repr__", []( dip::DataType const& a ) { return std::string( "PyDIP.DT_" ) + a.Name(); } );

   m.attr( "DT_BIN" ) = dip::DT_BIN;
   m.attr( "DT_UINT8" ) = dip::DT_UINT8;
   m.attr( "DT_SINT8" ) = dip::DT_SINT8;
   m.attr( "DT_UINT16" ) = dip::DT_UINT16;
   m.attr( "DT_SINT16" ) = dip::DT_SINT16;
   m.attr( "DT_UINT32" ) = dip::DT_UINT32;
   m.attr( "DT_SINT32" ) = dip::DT_SINT32;
   m.attr( "DT_SFLOAT" ) = dip::DT_SFLOAT;
   m.attr( "DT_DFLOAT" ) = dip::DT_DFLOAT;
   m.attr( "DT_SCOMPLEX" ) = dip::DT_SCOMPLEX;
   m.attr( "DT_DCOMPLEX" ) = dip::DT_DCOMPLEX;

   auto range = py::class_< dip::Range >( m, "Range" );
   range.def( py::init< dip::sint, dip::sint, dip::uint >(), "start"_a, "stop"_a, "step"_a = 1 );
   range.def( "__init__", []( dip::Range& self ) { new( &self ) dip::Range(); } );
   range.def( "__init__", []( dip::Range& self, dip::sint i ) { new( &self ) dip::Range( i ); }, "index"_a );
   range.def( "__repr__", []( dip::Range const& a ) {
      return std::string( "Range(" ) + std::to_string( a.start ) + "," + std::to_string( a.stop ) + "," + std::to_string( a.step ) + ")";
   } );

   init_image( m );
   init_display( m );
   init_math( m );
   init_statistics( m );
   init_linear( m );
   init_morphology( m );

   return m.ptr();
}
