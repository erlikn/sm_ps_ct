#!/bin/sh
((i=2000))
((k=0))
while [ $k -lt 20001 ]
do
	((k=i))
	echo $k
	python3 evaluator1.py mobilenet_rg0_1 $k
	((i=i+500))
done
