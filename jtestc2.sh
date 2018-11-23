#!/bin/sh
((i=3500))
((k=0))
while [ $k -lt 20000 ]
do
	((k=i))
	echo $k
	python3 tevaluator.py 181114c2 $k
	((i=i+50))
done
