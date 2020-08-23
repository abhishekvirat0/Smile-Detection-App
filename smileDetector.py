import cv2

# Face classifer
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# smile classifier
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# eye classifier
eye_detector = cv2.CascadeClassifier('haarcascade_Eye.xml')

# capturing face from webcam value can be anything -- can be a filename or 0 for webcam
webcam = cv2.VideoCapture(0)

while True:

    # Read current frame from webcam video stream
    successful_frame_read, frame = webcam.read()

    # if there's an error, abort
    if not successful_frame_read:
        break

    # color to grayscale i.e colorful to gray for optimization, recognization is not hindered in black and white, and rgb has 3(4 including briteness) channels but black and white has 1 channels.
    # color to gray i.e bgr i.e rgb backwards to gray 
    
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect where all the faces are, return list of co-oordinates, detect faces of all scale. faces here is list of list
    faces = face_detector.detectMultiScale(frame_grayscale)

    # # detect smile, scalefactor -> how much blur you wanna do image blur image (optimization), minNeighbors ->  min neighbor should be 20 to consider it as a smile
    # smiles = smile_detector.detectMultiScale(frame_grayscale, scaleFactor = 1.7, minNeighbors = 20)

    # run face detection within each faces
    for (x,y,w,h) in faces:
        # draw a  rectangle around the face
        cv2.rectangle(frame,(x,y), (x+w,y +h), (100,200,50), 4)

        # get this square image instead of all frame

        # get the sub frame ( using numpy.array(list_name)[0:2][1:3]  but since oprncv is build on numpy so everything here is numpy array)
        #(using numpy N-dimentional array slicing)
        # the_face = (x,y,w,h) just getting this portion of frame
        the_face = frame[y: y + h, x: x + w]

        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # detect smile, scalefactor -> how much blur you wanna do image blur image (optimization), minNeighbors ->  min neighbor should be 20 to consider it as a smile
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor = 1.7, minNeighbors = 20)

        eyes = eye_detector.detectMultiScale(face_grayscale,  scaleFactor = 1.1, minNeighbors = 20)



        
        # find all smile in face and draw all the rectangle around smile
        
        for (x_,y_,w_,h_) in smiles:
            
            ## draw rectangle around the smile
            cv2.rectangle(the_face,(x_,y_), (x_ + w,y_ + h), (50,50,200), 4)

        for (x_,y_,w_,h_) in eyes:
            
            ## draw rectangle around the smile
            cv2.rectangle(the_face,(x_,y_), (x_ + 20,y_ + 20), (255,255,255), 4)

        # label this face as smiling

        if(len(smiles) > 0):
            cv2.putText(frame, 'smiling', (x, y +h + 40), fontScale=3, fontFace = cv2.FONT_HERSHEY_PLAIN, color= (255,255,255))
        
        # if(len(eyes) > 0):
            # cv2.putText(frame, 'eyes identified', (x, y +h + 90), fontScale=3, fontFace = cv2.FONT_HERSHEY_PLAIN, color= (255,255,255))

    # run smile detection within each faces
    # for (x,y,w,h) in smiles:
        
    #     # draw a  rectangle around the face
    #     cv2.rectangle(frame,(x,y), (x+w,y +h), (50,50,200), 4)

    # print(faces)

    # window name i.e here it is 'Smile Detector' and show current frame
    cv2.imshow('Smile Detector', frame)

    # Display and wait until key is pressed, if we add 1 inside waitKey, it waits for 1 millisecond
    cv2.waitKey(1)

# cleanUp
# releases webcam from task manager so other application can use
webcam.release()

# closes all windows i.e make sure no window remains open
cv2.destroAllWindows()
print("Code Completed! ")
