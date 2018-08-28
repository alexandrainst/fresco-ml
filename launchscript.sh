#!/bin/sh

if test "$#" -ne 2; then
    echo 'Illegal number of parameters'
    echo 'usage: ./launchscript.sh [number of parties] [number of local examples]'
    exit 1
fi

echo Running MNIST FL Demo with $1 parties each with $2 local examples
if test $1 -ne 1; then 
    for i in `seq 2 $1`;
    do
        echo Starting party $i
        java -jar target/fresco-ml-0.0.1-SNAPSHOT-jar-with-dependencies.jar -i $i -n $1 -l $2 &>/dev/null &
    done
fi
echo Starting party 1
java -jar target/fresco-ml-0.0.1-SNAPSHOT-jar-with-dependencies.jar -i 1 -n $1 -l $2
echo Ran MNIST FL Demo with $1 parties each with $2 local examples
