#!/bin/sh
((i=3000))
((k=0))
while [ $k -lt 21000 ]
do
	((k=i))
	echo $k
	python3 evaluator.py 181114c2 $k
	((i=i+250))
done
