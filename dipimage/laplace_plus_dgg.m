%LAPLACE_PLUS_DGG   Laplace + Dgg
%  The zero-crossings of the result correspond to the edges in the image,
%  just as they do for the individual Laplace and Dgg operators. However,
%  the localization is improved by an order of magnitude with respect to
%  the individual operators.
%
% SYNOPSIS:
%  image_out = laplace_plus_dgg(image_in,sigma,method,boundary_condition,process,truncation)
%
%  IMAGE_IN is a scalar image with N dimensions.
%  IMAGE_OUT is a scalar image.
%
%  PROCESS determines along which dimensions to take the derivative.
%
%  See DERIVATIVE for a description of the parameters and the defaults.
%
% DIPlib:
%  This function calls the DIPlib function dip::LaplaceMinusDgg.

% (c)2018, Cris Luengo.
% Based on original DIPlib code: (c)1995-2014, Delft University of Technology.
% Based on original DIPimage code: (c)1999-2014, Delft University of Technology.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%    http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function out = laplace_plus_dgg(varargin)
out = compute_derivatives('laplace_plus_dgg',varargin{:});
