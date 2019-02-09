#!/bin/sh
((i=10000))
((k=0))
while [ $k -lt 80001 ]
do
	((k=i))
	echo $k
	python3 evaluator.py incResNetV2 $k
	((i=i+500))
done
