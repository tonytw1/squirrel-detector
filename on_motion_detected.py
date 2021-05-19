
#!/usr/bin/python
import sys
import os

message = ""
for arg in sys.argv[1:]:
        message += arg + ":"

os.system("mosquitto_pub -t motion -m " + message)
