import cv2 as cv
import mediapipe as mp
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import keyboard

def angle_calculate(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    
    radians=np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle=360-angle
    
    return int(angle)   

def image_process (frame,mp_drawing,mp_holistic,holistic):  
    angle = 0
    #cambios de color y aplicar módulo holistic
    image= cv.cvtColor(frame,cv.COLOR_RGB2BGR)
    result=holistic.process(image)
    image= cv.cvtColor(image,cv.COLOR_BGR2RGB)
        #Landmarks
    try: 
        landmarks = result.pose_landmarks.landmark
        
        #coordenadas de brazo izq
        shoulder_L = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                  landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow_L = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
        wrist_L = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
        
       
        angle = angle_calculate(shoulder_L,elbow_L,wrist_L)
        #look angle
        cv.putText(image,str(angle),
                   tuple(np.multiply(elbow_L,[640,480]).astype(int)),
                         cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv.LINE_AA)
        
    except:
        pass
     #dibujar las articulaciones del cuerpo en la imagen
    mp_drawing.draw_landmarks(image, result.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color = (102,31,208),thickness = 2,circle_radius = 2),
                              mp_drawing.DrawingSpec(color = (103,249,237),thickness = 2,circle_radius = 2))
    return image,angle
    
    

def Mistake_calculation(y_true,y_pred):
    y_true.pop(0)
    y_pred.pop(0)
    mse = mean_squared_error(y_true, y_pred)
    print(y_pred)

    print(mse)

    plt.title("Erro Cuadrático Medio")
    plt.plot(y_true,'r',label="Referencia")
    plt.plot(y_pred,'bo',label="Datos obtenidos")
    plt.ylabel('Angle')
    
    
def main(): 
    y_true = [0,30,45,60,75,90,105,120,135,150]
    y_pred = []
    i= 0  
    run = True
    #setup mediapie
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    
    #Abrir cámara web 
    capture = cv.VideoCapture(0)
    print(y_true[i])    
    with mp_holistic.Holistic(min_detection_confidence=0.8,min_tracking_confidence=0.8)as holistic:
        while run:
            #Lerr datos de camara web
            data,frame = capture.read()
            frame=cv.flip(frame,1)
            image,angle=image_process(frame,mp_drawing,mp_holistic,holistic)
            cv.imshow('camera',image)
            
            if cv.waitKey(1) == ord('q'):
                break 
            if keyboard.is_pressed('m'):
                list.append(y_pred,angle)
                i+=1
                if i <= 9:
                    print(y_true[i])
                else:
                    run= False
    
    cv.destroyAllWindows()
    capture.release()  
    Mistake_calculation(y_true, y_pred)

if __name__=="__main__":
    main()          
        
