#!/bin/sh
((i=2000))
((k=0))
while [ $k -lt 20000 ]
do
	((k=i))
	echo $k
	python3 tevaluator.py 181114rgmS5 $k
	((i=i+50))
done
