#!/bin/sh
((i=2000))
((k=0))
while [ $k -lt 20000 ]
do
	((k=i))
	echo $k
	python3 evaluator.py 181114rg0_deconv $k
	((i=i+50))
done
