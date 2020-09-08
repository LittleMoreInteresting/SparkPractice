import time
import os

outputFile = "sub_" + time.strftime("%Y-%m-%d-%H-%M", time.localtime()) + ".csv"
fpOut = open(outputFile, "w")
fpOut.writelines(["WindNumber,Time,label\n"])

for n in range(1, 13):
    inputPath = "E:/myPractice/result_%s" % n + "/"
    if not os.path.exists(inputPath):
        continue
    dirList = os.listdir(inputPath)
    print(inputPath)
    for file in dirList:
        if file.endswith(".csv"):
            fpIn = open(inputPath+file, "r")
            fpOut.writelines(fpIn.readlines())
            print(file)
