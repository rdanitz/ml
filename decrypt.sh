#!/bin/sh

# $1: .asc file 
function dec {
  dir=`echo $1 | sed -e 's/\/[^\/]*$//g'`
  f=`echo $1 | sed -e 's/.asc$//g'`
  openssl aes-128-cbc -d -nosalt -base64 -pass file:$dir/secret.txt -in $1 -out $f
}

if [ $# -eq 0 ]
then
  for f in */*.jl.asc
  do
    dec $f 
  done
else
  for f in $1/*.jl.asc
  do
    dec $f
  done
fi
