#!/bin/sh

path=$1
locpath=$2

unpack_path=/tmp/$locpath

mkdir -p $unpack_path
if [ -f $path ]; then
  #echo "Copying $path to $unpack_path"
  cp $path $unpack_path
  cwd=$(pwd)
  cd $unpack_path
  #echo $(pwd)
  filename=$(basename $path)
  #echo $filename
  if [ -f $filename ]; then
    #echo running tar xvf $filename
    tar xvf $filename > dump
    dir_name=$(head -n 1 dump)
    #dir_name=$(tar xvf $filename | head -n 1)
    #cat dump
    rm -f dump
    cd $cwd
    echo $unpack_path/$dir_name
  else
    echo "$filename does not exist"
  fi
else
  echo "$path does not exist"
fi
