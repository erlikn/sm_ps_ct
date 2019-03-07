#!/bin/sh
((i=2000))
((k=0))
while [ $k -lt 20000 ]
do
	((k=i))
	echo $k
	python3 tevaluator.py 181114rg0_ohem $k
	((i=i+50))
done
