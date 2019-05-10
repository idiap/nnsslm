#!/bin/bash
#################################################################################
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/             #
# Written by Weipeng He <weipeng.he@idiap.ch>                                   #
#                                                                               #
# This file is part of "Neural Network based Sound Source Localization Models". #
#                                                                               #
# "Neural Network based Sound Source Localization Models" is free software:     #
# you can redistribute it and/or modify it under the terms of the BSD 3-Clause  #
# License.                                                                      #
#                                                                               #
# "Neural Network based Sound Source Localization Models" is distributed in     #
# the hope that it will be useful, but WITHOUT ANY WARRANTY; without even       #
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR           #
# PURPOSE. See the BSD 3-Clause License for more details.                       #
#################################################################################

usage(){
  >&2 echo "Usage: $0 EXECUTABLE DATA_DIR FEATURE_NAME [OPTIONS]"
  exit 1
}

on_error(){
  >&2 echo "Error: failed to process $1"
  exit 2
}

# parse arguments
prog=$1
ddir=$2
name=$3
optn=$4

[ -x "$prog" ] || usage
[ -d "$ddir" ] || usage
[ -n "$name" ] || usage

dest="${ddir}/features/${name}"
mkdir -p $dest || on_error 'mkdir'
for x in ${ddir}/data/*.wav
do
  echo $x
  $prog $optn $x $dest || on_error $x
done

