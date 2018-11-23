#!/bin/sh
((i=2000))
((k=0))
while [ $k -lt 21000 ]
do
	((k=i))
	echo $k
	python3 evaluator.py 181114rg0 $k
	((i=i+50))
done
