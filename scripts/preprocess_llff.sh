#!/bin/bash

# Get the path of the script
SCRIPT_PATH="$(realpath $0)"

# Get the directory of the script
SCRIPT_DIR="$(dirname $SCRIPT_PATH)"

for d in $(ls .)
do
	cd $d
	python $SCRIPT_DIR/llff2nerf.py --images images .
	cd ..
done
