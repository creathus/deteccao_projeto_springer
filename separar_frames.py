import cv2

cap1 = cv2.VideoCapture("/home/cesar/Downloads/fhd_c_ilum.mp4")

i = 0
while True:
    success1, frame1 = cap1.read()

    if success1:

        if i % 1 == 0:
            print(i)
            cv2.imwrite("/home/cesar/data/images/"+str(i)+"_frame.jpg", frame1)
        i = i + 1
        print(i)