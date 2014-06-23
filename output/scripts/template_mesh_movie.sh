#!/bin/bash

### USER CONFIGURATION (USER ACCESIBLE) ###

MESH_TYPE=field
INI=999
FIN=200000
INC=1000

### SCRIPT CONFIGURATION (NOT USER ACCESIBLE) ###

FILENAME="$MESH_TYPE"_movie 

### GENERATION OF GNUPLOT SCRIPT ###

echo set terminal jpeg size 1280,720 >> plot_"$MESH_TYPE".gpi

echo set nokey >> plot_"$MESH_TYPE".gpi
echo set grid >> plot_"$MESH_TYPE".gpi
echo set ylabel \""$MESH_TYPE" \(simulation units\)\" >> plot_"$MESH_TYPE".gpi
echo set xlabel \"node \(ds = 10% debye lenght\)\" >> plot_"$MESH_TYPE".gpi
#echo set xrange [0:12.7] >> plot_"$MESH_TYPE".gpi
#echo set yrange [0:102.1] >> plot_"$MESH_TYPE".gpi

echo j=0 >> plot_"$MESH_TYPE".gpi

echo do for[i=$INI:$FIN/$INC] \{ >> plot_"$MESH_TYPE".gpi

echo imod = i*$INC-1 >> plot_"$MESH_TYPE".gpi
echo set output \""$FILENAME"_\".j.\".jpg\" >> plot_"$MESH_TYPE".gpi
echo -e set title \""$MESH_TYPE" \(t = \".imod.\"\)\" >> plot_"$MESH_TYPE".gpi
echo -e plot \'./"$MESH_TYPE"_t_\'.imod.\'.dat\' >> plot_"$MESH_TYPE".gpi

echo j=j+1 >> plot_"$MESH_TYPE".gpi

echo \} >> plot_"$MESH_TYPE".gpi

### EXECUTE GNUPLOT SCRIPT FOR FRAMES GENERATION ###

gnuplot plot_"$MESH_TYPE".gpi

### REMOVE GNUPLOT SCRIPT ###

rm plot_"$MESH_TYPE".gpi

### GENERATE MOVIE FROM FRAMES AND REMOVE FRAMES ###

avconv -f image2 -i "$FILENAME"_%d.jpg -b 32000k "$FILENAME".mp4
find . -name '*.jpg' -type f -print -delete
