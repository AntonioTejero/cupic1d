#!/bin/bash

### USER CONFIGURATION (USER ACCESIBLE) ###

PARTICLE_TIPE=ion
INI=0
FIN=1000000
INC=10

### SCRIPT CONFIGURATION (NOT USER ACCESIBLE) ###

FILENAME="$PARTICLE_TIPE"_movie 
NUMBER_OF_BINS=100

### GENERATION OF GNUPLOT SCRIPT ###

echo set terminal jpeg size 1280,720 >> plot.gpi
echo bin\(x,width\) = width*floor\(x/width\) >> plot.gpi

echo j=0 >> plot.gpi
echo do for[i=$INI:$FIN:$INC] \{ >> plot.gpi
  
echo set output \""$FILENAME"_\".j.\".jpg\" >> plot.gpi
echo set nokey >> plot.gpi
echo set size 1,1 >> plot.gpi
echo set origin 0,0 >> plot.gpi
echo set multiplot >> plot.gpi
echo set tmargin 0.5 >> plot.gpi
echo set bmargin 0.5 >> plot.gpi
echo set rmargin 1.0 >> plot.gpi
echo set lmargin 5.0 >> plot.gpi
echo set grid >> plot.gpi

#echo set xrange [0:12.7] >> plot.gpi
#echo set yrange [0:102.1] >> plot.gpi
echo set ylabel \"number of "$PARTICLE_TIPE"\" >> plot.gpi
echo set xlabel \"r\" >> plot.gpi
echo set size 0.45,0.85 >> plot.gpi
echo set origin 0.03,0.07 >> plot.gpi
echo unset key >> plot.gpi
echo unset ytics >> plot.gpi
echo -e set title \""$PARTICLE_TIPE" position distribution \(t = \".i.\"\)\" >> plot.gpi
echo -e stats \'./"$PARTICLE_TIPE"s_t_\'.i.\'.dat\' u 1 nooutput >> plot.gpi
echo -e plot \'./"$PARTICLE_TIPE"s_t_\'.i.\'.dat\' u \(bin\(\$1, \(STATS_max-STATS_min\)/$NUMBER_OF_BINS\)\):\(1.0\) smooth freq with boxes lc rgb \"blue\" >> plot.gpi

#echo set xrange [0:12.7] >> plot.gpi
#echo set yrange [0:102.1] >> plot.gpi
echo set ylabel \"frequency \" >> plot.gpi
echo set xlabel \"velocity\" >> plot.gpi
echo unset ytics >> plot.gpi
echo set size 0.45,0.85 >> plot.gpi
echo set origin 0.51,0.07 >> plot.gpi
echo -e set title \""$PARTICLE_TIPE" velocity distribution \(t = \".i.\"\)\" >> plot.gpi
echo -e stats \'./"$PARTICLE_TIPE"s_t_\'.i.\'.dat\' u 2 nooutput >> plot.gpi
echo -e plot \'./"$PARTICLE_TIPE"s_t_\'.i.\'.dat\' u \(bin\(\$2, \(STATS_max-STATS_min\)/$NUMBER_OF_BINS\)\):\(1.0\) smooth freq with boxes lc rgb \"red\" >> plot.gpi
echo set ytics >> plot.gpi

echo unset multiplot >> plot.gpi
echo j=j+1 >> plot.gpi

echo \} >> plot.gpi

### EXECUTE GNUPLOT SCRIPT FOR FRAMES GENERATION ###

gnuplot plot.gpi

### REMOVE GNUPLOT SCRIPT ###

rm plot.gpi

### GENERATE MOVIE FROM FRAMES AND REMOVE FRAMES ###

avconv -f image2 -i "$FILENAME"_%d.jpg -b 32000k "$FILENAME".mp4
find . -name '*.jpg' -type f -print -delete
