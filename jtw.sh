#!/bin/sh
((i=0))
while :
do
	((k=i+18000))
	echo $k
	python3 evaluator.py 180916c2 $k
	((i=i+1000))
done
