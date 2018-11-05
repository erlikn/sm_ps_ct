#!/bin/sh
((i=0))
while :
do
	((k=i+10000))
	echo $k
	python3 evaluator1.py 180912c2new $k
	((i=i+1000))
done
