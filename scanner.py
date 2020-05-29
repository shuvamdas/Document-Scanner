import cv2
import numpy as np
import mapper
image=cv2.imread("test_img.jpg")   
image=cv2.resize(image,(1300,800)) #resizng because opencv does not work well with bigger images
orig=image.copy()

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
cv2.imshow("gray",gray)

blurred=cv2.GaussianBlur(gray,(5,5),0)  #(5,5)=kernel size and 0=sigma that determines the amount of blur
cv2.imshow("Blur",blurred)

edged=cv2.Canny(blurred,30,50)  #30 minthreshold and 50 is the maxthreshold
cv2.imshow("Canny",edged)


contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  #retrieve the contours as a list
contours=sorted(contours,key=cv2.contourArea,reverse=True)

#loop extracts the boundary contours of the page
for c in contours:
    p=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*p,True)

    if len(approx)==4:
        target=approx
        break
approx=mapper.mapp(target) #find endpoints of the sheet

pts=np.float32([[0,0],[800,0],[800,400],[0,400]])  #map to 800*400 target window

op=cv2.getPerspectiveTransform(approx,pts)  #get top or bird eye view effect
dst=cv2.warpPerspective(orig,op,(800,400))


cv2.imshow("Scanned",dst)
