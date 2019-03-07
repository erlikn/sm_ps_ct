#!/bin/sh
((i=9750))
((k=0))
while [ $k -lt 9750 ]
do
	((k=i))
	echo $k
	python3 evaluator_deconv.py 181114rg0_deconv $k
	((i=i+50))
done
