#!/bin/sh

path=$1
unpack_path=/tmp

cp $path $unpack_path
cwd=$(pwd)
cd $unpack_path
#echo $(pwd)
filename=$(basename $path)
#echo $filename
#echo running tar xvf $filename
tar xvf $filename > dump
dir_name=$(head -n 1 dump)
#dir_name=$(tar xvf $filename | head -n 1)
rm dump
cd $cwd
echo $unpack_path/$dir_name
