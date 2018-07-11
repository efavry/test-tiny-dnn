#!/bin/sh

path=$1

cp $path /tmp
cwd=$(pwd)
cd /tmp
#echo $(pwd)
filename=$(basename $path)
#echo $filename
#echo running tar xvf $filename
dir_name=$(tar xvf $filename | head -n 1)
cd $cwd
echo /tmp/$dir_name
