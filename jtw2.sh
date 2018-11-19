#!/bin/sh
((i=0))
((k=0))
while [ $k -lt 45000 ]
do
	((k=i+1000))
	echo $k
	python3 evaluator.py 181114c2 $k
	((i=i+1000))
done
