#!/bin/sh
((i=2000))
((k=0))
while [ $k -lt 20000 ]
do
	((k=i))
	echo $k
	python3 tevaluator_gpu1.py 181114rg0 $k
	((i=i+50))
done
