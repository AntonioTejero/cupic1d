#!/bin/bash

### USER CONFIGURATION (USER ACCESIBLE) ###

MESH_TYPE=field
INI=9
FIN=200000
INC=100

### SCRIPT CONFIGURATION (NOT USER ACCESIBLE) ###

FILENAME="$MESH_TIPE"_movie 
NUMBER_OF_BINS=100

### GENERATION OF GNUPLOT SCRIPT ###

echo set terminal jpeg size 1280,720 >> plot.gpi

echo set nokey >> plot.gpi
echo set grid >> plot.gpi
echo set ylabel \""$MESH_TYPE" \(simulation units\)\" >> plot.gpi
echo set xlabel \"node \(ds = 10% debye lenght\)\" >> plot.gpi
#echo set xrange [0:12.7] >> plot.gpi
#echo set yrange [0:102.1] >> plot.gpi

echo j=0 >> plot.gpi

echo do for[i=$INI:$FIN:$INC] \{ >> plot.gpi

echo set output \""$FILENAME"_\".j.\".jpg\" >> plot.gpi
echo -e set title \""$MESH_TYPE" \(t = \".i.\"\)\" >> plot.gpi
echo -e plot \'./"$MESH_TYPE"_t_\'.i.\'.dat\' >> plot.gpi

echo j=j+1 >> plot.gpi

echo \} >> plot.gpi

### EXECUTE GNUPLOT SCRIPT FOR FRAMES GENERATION ###

gnuplot plot.gpi

### REMOVE GNUPLOT SCRIPT ###

rm plot.gpi

### GENERATE MOVIE FROM FRAMES AND REMOVE FRAMES ###

avconv -f image2 -i "$FILENAME"_%d.jpg -b 32000k "$FILENAME".mp4
find . -name '*.jpg' -type f -print -delete
