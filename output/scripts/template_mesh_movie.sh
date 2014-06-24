#!/bin/bash

### USER CONFIGURATION (USER ACCESIBLE) ###

MESH_TYPE=field
AVERAGED=true
INI=1000
FIN=200000
INC=1000

### SCRIPT CONFIGURATION (NOT USER ACCESIBLE) ###

if [ "$AVERAGED" = true ] ; then
  OFILENAME=avg_"$MESH_TYPE"_movie 
  IFILENAME=avg_"$MESH_TYPE"
  TITLE="averaged $MESH_TYPE over $INC time steps"
  YLABEL="$MESH_TYPE \(simulation units\)"
  XLABEL="node \(ds = 10% debye lenght\)"
else
  OFILENAME="$MESH_TYPE"_movie 
  IFILENAME="$MESH_TYPE"
  TITLE="$MESH_TYPE over $INC time steps"
  YLABEL="$MESH_TYPE \(simulation units\)"
  XLABEL="node \(ds = 10% debye lenght\)"
fi

### GENERATION OF GNUPLOT SCRIPT ###

echo set terminal jpeg size 1280,720 >> plot_"$MESH_TYPE".gpi

echo set nokey >> plot_"$MESH_TYPE".gpi
echo set grid >> plot_"$MESH_TYPE".gpi
echo set ylabel \""$YLABEL"\" >> plot_"$MESH_TYPE".gpi
echo set xlabel \""$XLABEL"\" >> plot_"$MESH_TYPE".gpi
#echo set xrange [0:12.7] >> plot_"$MESH_TYPE".gpi
#echo set yrange [0:102.1] >> plot_"$MESH_TYPE".gpi

echo j=0 >> plot_"$MESH_TYPE".gpi

echo do for[i=$INI/$INC:$FIN/$INC] \{ >> plot_"$MESH_TYPE".gpi

echo imod = i*$INC-1 >> plot_"$MESH_TYPE".gpi
echo set output \""$OFILENAME"_\".j.\".jpg\" >> plot_"$MESH_TYPE".gpi
echo -e set title \""$TITLE" \(t = \".imod.\"\)\" >> plot_"$MESH_TYPE".gpi
echo -e plot \'./"$IFILENAME"_t_\'.imod.\'.dat\' >> plot_"$MESH_TYPE".gpi

echo j=j+1 >> plot_"$MESH_TYPE".gpi

echo \} >> plot_"$MESH_TYPE".gpi

### EXECUTE GNUPLOT SCRIPT FOR FRAMES GENERATION ###

gnuplot plot_"$MESH_TYPE".gpi

### REMOVE GNUPLOT SCRIPT ###

rm plot_"$MESH_TYPE".gpi

### GENERATE MOVIE FROM FRAMES AND REMOVE FRAMES ###

avconv -f image2 -i "$OFILENAME"_%d.jpg -b 32000k "$OFILENAME".mov
find . -name '*.jpg' -type f -print -delete
