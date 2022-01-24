#!/usr/bin/env bash

./build.sh

docker save pdacdetectioncontainer | gzip -c > ../PDAC_detection.tar.gz
