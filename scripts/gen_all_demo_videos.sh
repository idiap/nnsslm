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
  >&2 echo "Usage: $0 DBDIR OUTDIR"
  exit 1
}

on_error(){
  >&2 echo "Error: failed to process $1"
  exit 2
}

ddir=$1
odir=$2

[ -d "$ddir" ] || usage
[ -d "$odir" ] || mkdir -p $odir || on_error 'mkdir' 
script=$(mktemp) || on_error 'mktemp'

for x in ${ddir}/data/*.wav
do
  echo $x
  sid=${x%.*}
  sid=${sid##*/}
  dest="${odir}/${sid}"
  ../common/gen_demo_video.py -m rec_gccfb8192_rfcts_s1h500x2_s2h500_os_bn_sig8 -s ${sid} -w 8192 -o 4096 --min-score=0.6 $ddir $dest > $script || on_error $x
  xscript=$(tail -n+2 $script | sed 's/\\//g')
  eval $xscript || on_error $x
done
