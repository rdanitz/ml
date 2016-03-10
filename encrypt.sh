#!/bin/sh

# $1: source file 
function enc {
  dir=`echo $1 | sed -e 's/\/[^\/]*$//g'`
  openssl aes-128-cbc -nosalt -base64 -pass file:secret.txt -in $1 -out $1.asc
}

if [ $# -eq 0 ]
then
  for f in */*.jl
  do
    enc $f 
  done
else
  for f in $1/*.jl 
  do
    enc $f
  done
fi
