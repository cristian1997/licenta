#!/bin/bash

dataset=$1

if [ $# -lt 2 ]; then
	nr_iter=10
else
	nr_iter=$2
fi

#echo $nr_iter
mkdir -p out/$dataset-$nr_iter
./main $dataset $nr_iter
