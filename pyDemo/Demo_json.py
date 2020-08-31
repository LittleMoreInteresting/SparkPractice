import json
import time

data = {
    "name": "AA",
    "Age": 20,
    "no": 1
}

dumps = json.dumps(data)

data2 = json.loads(dumps)
print(dumps)
print(data2)

strTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(strTime)
print(time.mktime(time.strptime(strTime, "%Y-%m-%d %H:%M:%S")))
