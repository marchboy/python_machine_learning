# -*- coding: utf-8 -*-
# https://pypi.org/project/SpeechRecognition/


import speech_recognition as sr
print(sr.__version__)

record = sr.Recognizer()

harvard = sr.AudioFile("E:/0002_Proj/车载语义分析/public/1.wav")
with harvard as source:
    audio = record.record(harvard)
    print(type(audio))
    string = record.recognize_sphinx(audio)
    print(string)


