
from playsound import playsound
Line_Position2=8
result=[]
def drawboxtosafeline(a,b):
    distance_from_line = res = tuple(map(lambda i, j: i - j, a, b))
    res = sum(list(distance_from_line))
    if abs(res)<400 and abs(res)>0:
        playsound(r"D:\Projects\CarDistanceAlert\utils\alert.wav")

