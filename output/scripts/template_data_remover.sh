#!/bin/bash

FILENAME=ions_t_
INI=200010
FIN=300000
INC=10

for ((i=$INI; i<=$FIN; i=$i+$INC))
do
  rm "$FILENAME"$i.dat
  #find . -name '"$FILENAME"$i.dat' -type f -print -delete
done

