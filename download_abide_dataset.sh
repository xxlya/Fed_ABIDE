#!/bin/bash

for site in UM_1 NYU USM UCLA_1
do
  python3 download_abide_preproc.py -d rois_ho -p cpac -s filt_noglobal -t "$site" -o .
  mkdir "$site" "$site"_correlation_matrix
  mv Outputs/cpac/filt_noglobal/rois_ho/"$site"* "$site"/
done

rm -r Outputs
