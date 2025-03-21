from gpiozero import DistanceSensor 
import tkinter as tk 
from tkinter import font  
from time import sleep  

sensor1 = DistanceSensor(echo=23, trigger=24, max_distance=10)
sensor2 = DistanceSensor(echo=17, trigger=27, max_distance=10)
distanceBound = 5

window = tk.Tk()
window.title("Distance Measurement")
customFont = font.Font(size=30) 
window.geometry("800x400") 

distanceLabel1 = tk.Label(window, text="Distance: ", anchor='center', font=customFont)
distanceLabel2 = tk.Label(window, text="Distance: ", anchor='center', font=customFont)
distanceLabel3 = tk.Label(window, text="Posture: ", anchor='center', font=customFont)

distanceLabel1.pack()
distanceLabel2.pack() 
distanceLabel3.pack()

def isPostureCorrect(distance1, distance2):
    return abs(distance1 - distance2) < distanceBound

def measureDistance():
    distance1 = int(sensor1.distance * 100)
    distance2 = int(sensor2.distance * 100) 
    
    distanceLabel1.config(fg="red", text="Distance: {} cm\n".format(distance1))
    distanceLabel2.config(fg="red", text="Distance: {} cm\n".format(distance2))
    distanceLabel3.config(fg="red", text="Posture: {} ".format(isPostureCorrect(distance1, distance2)))
    window.after(100, measureDistance)  

measureDistance()

window.mainloop()