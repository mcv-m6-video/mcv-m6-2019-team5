#!/usr/bin/env bash

cd pycocotools/PythonAPI
make
cd ../..

cd faster-rcnn
python setup.py build develop