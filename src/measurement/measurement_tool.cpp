/*
 * DIPlib 3.0
 * This file contains definitions for functions in the dip::MeasurementTool class.
 *
 * (c)2016, Cris Luengo.
 * Based on original DIPlib code: (c)1995-2014, Delft University of Technology.
 */

#include "diplib.h"
#include "diplib/measurement.h"
#include "diplib/framework.h"
#include "diplib/regions.h"

// FEATURES:
// Core
#include "feature_size.h"
#include "feature_perimeter.h"
#include "feature_surface_area.h"
#include "feature_feret.h"
// Moments
#include "feature_gravity.h"
// Shape features
#include "feature_p2a.h"


namespace dip {


MeasurementTool::MeasurementTool() {
   // Core
   Register( dip::Feature::Pointer( new Feature::FeatureSize ));
   Register( dip::Feature::Pointer( new Feature::FeaturePerimeter ));
   Register( dip::Feature::Pointer( new Feature::FeatureSurfaceArea ));
   Register( dip::Feature::Pointer( new Feature::FeatureFeret ));
   // Moments
   Register( dip::Feature::Pointer( new Feature::FeatureGravity ));
   // Shape features
   Register( dip::Feature::Pointer( new Feature::FeatureP2A ));

   // TODO: register all measurement features
   /*
   //DIPXJ( dip_FeatureSizeRegister( 0 ));
   DIPXJ( dip_FeatureDimensionRegister( 0 ));
   DIPXJ( dip_FeatureMinimumRegister( 0 ));
   DIPXJ( dip_FeatureMaximumRegister( 0 ));

   //DIPXJ( dip_FeaturePerimeterRegister( 0 ));
   //DIPXJ( dip_FeatureSurfaceAreaRegister( 0 ));
   //DIPXJ( dip_FeatureFeretRegister( 0 ));
   DIPXJ( dip_FeatureAspectRatioFeretRegister( 0 ));
   DIPXJ( dip_FeatureRadiusRegister( 0 ));
   DIPXJ( dip_FeatureConvexAreaRegister( 0 ));
   DIPXJ( dip_FeatureConvexPerimeterRegister( 0 ));

   //DIPXJ( dip_FeatureP2ARegister( 0 ));
   DIPXJ( dip_FeatureShapeRegister( 0 ));
   DIPXJ( dip_FeatureConvexityRegister( 0 ));

   DIPXJ( dip_FeatureSumRegister( 0 ));
   DIPXJ( dip_FeatureMeanRegister( 0 ));
   DIPXJ( dip_FeatureStdDevRegister( 0 ));
   DIPXJ( dip_FeatureSkewnessRegister( 0 ));
   DIPXJ( dip_FeatureExcessKurtosisRegister( 0 ));
   DIPXJ( dip_FeatureMaxValRegister( 0 ));
   DIPXJ( dip_FeatureMinValRegister( 0 ));

   DIPXJ( dip_FeatureCenterRegister( 0 ));
   //DIPXJ( dip_FeatureGravityRegister( 0 ));
   DIPXJ( dip_FeatureInertiaRegister( 0 ));
   DIPXJ( dip_FeatureGinertiaRegister( 0 ));
   DIPXJ( dip_FeatureMuRegister( 0 ));
   DIPXJ( dip_FeatureGmuRegister( 0 ));

   DIPXJ( dip_FeatureBendingEnergyRegister( 0 ));
   DIPXJ( dip_FeatureChainCodeBendingEnergyRegister( 0 ));
   DIPXJ( dip_FeatureAnisotropy2DRegister( 0 ));
   DIPXJ( dip_FeatureOrientation2DRegister( 0 ));
   DIPXJ( dip_FeatureLongestChaincodeRunRegister( 0 ));
   */
}

using LineBasedFeatureArray = std::vector< Feature::LineBased* >;
using FeatureArray = std::vector< Feature::Base* >;


// dip::Framework::ScanFilter function, not overloaded because the Feature::LineBased::Measure functions
// that we call here are not overloaded.
static void dip__Measure(
      std::vector< Framework::ScanBuffer > const& inBuffer,
      std::vector< Framework::ScanBuffer >& outBuffer,
      dip::uint bufferLength,
      dip::uint dimension,
      UnsignedArray const& position,
      void const* functionParameters,
      void*
) {
   LineIterator< uint32 > label(
         static_cast< uint32* >( inBuffer[ 0 ].buffer ),
         0, bufferLength, inBuffer[ 0 ].stride,
         inBuffer[ 0 ].tensorLength, inBuffer[ 0 ].tensorStride
   );
   LineIterator< dfloat > grey;
   if( inBuffer.size() > 1 ) {
      grey = LineIterator< dfloat >(
            static_cast< dfloat* >( inBuffer[ 1 ].buffer ),
            0, bufferLength, inBuffer[ 1 ].stride,
            inBuffer[ 1 ].tensorLength, inBuffer[ 1 ].tensorStride
      );
   }

   LineBasedFeatureArray const& features = *( static_cast< LineBasedFeatureArray const* >( functionParameters ));
   for( auto const& feature : features ) {
      feature->Measure( label, grey, position, dimension );
   }
}


Measurement MeasurementTool::Measure(
      Image const& label,
      Image const& grey,
      StringArray features, // copy
      UnsignedArray const& objectIDs,
      dip::uint connectivity
) const {

   // Check input
   DIP_THROW_IF( label.TensorElements() != 1, E::NOT_SCALAR );
   DIP_THROW_IF( !label.DataType().IsUnsigned(), E::DATA_TYPE_NOT_SUPPORTED );
   if( grey.IsForged() ) {
      DIP_THROW_IF( !grey.DataType().IsReal(), E::DATA_TYPE_NOT_SUPPORTED );
      DIP_TRY
         grey.CompareProperties( label, Option::CmpProps_Sizes );
      DIP_CATCH
   }

   // Parse the features array and prepare measurements
   DIP_THROW_IF( features.empty(), "No features given" );
   FeatureArray featureArray;
   featureArray.reserve( features.size() );
   Measurement measurement;
   dip::uint ii = 0;
   while( ii < features.size() ) {
      String const& name = features[ ii ];
      if( !measurement.FeatureExists( name ) ) {
         Feature::Base* feature = features_[ Index( name ) ].get();
         if( feature->information.needsGreyValue ) {
            DIP_THROW_IF( !grey.IsForged(), "Measurement feature requires grey-value image" );
         }
         featureArray.push_back( feature );
         DIP_TRY
            Feature::ValueInformationArray values = feature->Initialize( label, grey );
            measurement.AddFeature( name, values );
         DIP_CATCH
         if( feature->type == Feature::Type::COMPOSITE ) {
            StringArray names = dynamic_cast< Feature::Composite* >( feature )->Dependencies();
            for( auto const& n : names ) {
               features.push_back( n ); // Add features needed for composite measure to the list of features to process in this loop
            }
         }
      }
      ++ii;
   }

   // Fill out the object IDs
   if( objectIDs.empty() ) {
      measurement.AddObjectIDs( GetObjectLabels( label, Image(), false ));
   } else {
      measurement.AddObjectIDs( objectIDs );
   }

   // Allocate memory for all features and objects
   measurement.Forge();

   // Figure out which types of measurements we want to do
   bool doLineBased = false;
   LineBasedFeatureArray lineBasedFeatures;
   bool doImageBased = false;
   bool doChaincodeBased = false;
   bool doConvHullBased = false;
   bool doComposite = false;
   for( auto const& feature : featureArray ) {
      switch( feature->type ) {
         case Feature::Type::LINE_BASED:
            doLineBased = true;
            lineBasedFeatures.emplace_back( dynamic_cast< Feature::LineBased* >( feature ));
            break;
         case Feature::Type::IMAGE_BASED:
            doImageBased = true;
            break;
         case Feature::Type::CHAINCODE_BASED:
            doChaincodeBased = true;
            break;
         case Feature::Type::CONVEXHULL_BASED:
            doConvHullBased = true;
            break;
         case Feature::Type::COMPOSITE:
            doComposite = true;
            break;
      }
   }

   // Let the line based functions do their work
   if( doLineBased ) {

      // Create arrays for Scan framework
      ImageConstRefArray inar{ label };
      DataTypeArray inBufT{ DT_UINT32 };
      if( grey.IsForged() ) {
         inar.emplace_back( grey );
         inBufT.emplace_back( DT_DFLOAT );
      }
      ImageRefArray outar{};
      DataTypeArray outBufT{};
      DataTypeArray outImT{};
      UnsignedArray nElem{};

      // Do the scan, which calls dip::Feature::LineBased::Measure()
      Framework::Scan(
            inar, outar, inBufT, outBufT, outImT, nElem,
            dip__Measure,
            &lineBasedFeatures,
            {},
            Framework::Scan_NoMultiThreading + Framework::Scan_NeedCoordinates
      );

      // Call dip::Feature::LineBased::Finish()
      for( auto const& feature : lineBasedFeatures ) {
         Measurement::IteratorFeature column = measurement[ feature->information.name ];
         Measurement::IteratorFeature::Iterator it = column.FirstObject();
         do {
            feature->Finish( it.ObjectID(), it.data() );
         } while( ++it );
      }
   }

   // Let the image based functions do their work
   if( doImageBased ) {
      for( auto const& feature : featureArray ) {
         if( feature->type == Feature::Type::IMAGE_BASED ) {
            Measurement::IteratorFeature column = measurement[ feature->information.name ];
            dynamic_cast< Feature::ImageBased* >( feature )->Measure( label, grey, column );
         }
      }
   }

   // Let the chaincode based functions do their work
   if( doChaincodeBased || doConvHullBased ) {
      ChainCodeArray chainCodeArray = GetImageChainCodes( label, measurement.Objects(), connectivity );
      auto itCC = chainCodeArray.begin();
      auto itObj = measurement.FirstObject(); // these two arrays are ordered the same way
      do {
         ConvexHull ch;
         if( doConvHullBased ) {
            ch = itCC->ConvexHull();
         }
         for( auto const& feature : featureArray ) {
            if( feature->type == Feature::Type::CHAINCODE_BASED ) {
               auto cell = itObj[ feature->information.name ];
               dynamic_cast< Feature::ChainCodeBased* >( feature )->Measure( *itCC, cell.data() );
            } else if( feature->type == Feature::Type::CONVEXHULL_BASED ) {
               auto cell = itObj[ feature->information.name ];
               dynamic_cast< Feature::ConvexHullBased* >( feature )->Measure( ch, cell.data() );
            }
         }
      } while( ++itCC, ++itObj );
   }

   // Let the composite functions do their work
   if( doComposite ) {
      Measurement::IteratorObject row = measurement.FirstObject(); // these two arrays are ordered the same way
      do {
         for( auto const& feature : featureArray ) {
            if( feature->type == Feature::Type::COMPOSITE ) {
               auto cell = row[ feature->information.name ];
               dynamic_cast< Feature::Composite* >( feature )->Measure( row, cell.data() );
            }
         }
      } while( ++row );
   }

   // Clean up
   for( auto const& feature : featureArray ) {
      feature->Cleanup();
   }

   return measurement;
}


std::ostream& operator<<(
      std::ostream& os,
      Measurement const& msr
) {
   // TODO
   return os;
}


} // namespace dip
