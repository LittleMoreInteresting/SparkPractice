import time
import os
inputPath = "E:/myPractice/result.csv/"
outputFile = "sub_" + time.strftime("%Y-%m-%d-%H-%M", time.localtime()) + ".csv"
fpOut = open(outputFile, "w")
fpOut.writelines(["WindNumber,Time,label\n"])
dirList = os.listdir(inputPath)
for file in dirList:
    if file.endswith(".csv"):
        fpIn = open(inputPath+file, "r")
        fpOut.writelines(fpIn.readlines())
        print(file)
