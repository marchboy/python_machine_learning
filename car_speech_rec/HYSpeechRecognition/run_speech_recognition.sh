#!/bin/sh


date+'%Y%m%d-%T'

file_path='/home/flamingo/car_speech_recognition/'
echo "run_speech_recognition.sh"
echo "run speech_recognition_server.py"

/usr/bin/python -u ${file_path}speech_recognition_server.py

date+'%Y%m%d-%T'

echo "--------------------------------------------"

