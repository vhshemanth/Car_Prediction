
from playsound import playsound

def drawboxtosafeline(a,b):
    if len(a)>0 and len(b)>0:
        distance_from_line = tuple(map(lambda i, j: i - j, a, b))
        res = sum(list(distance_from_line))
        print(res)
        if abs(res)<400 and abs(res)>300:
            playsound(r"D:\Projects\CarDistanceAlert\utils\alert.wav")
            # pass

