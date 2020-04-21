import numpy as np
import cv2
import os

# imgpath= "D:\\BaiduNetdiskDownload\\water_dataset\\training\\images\\1\\2\\"
imgpath= "D:\\BaiduNetdiskDownload\\water_dataset\\test\\test-data\\1\\2\\"
filelist= os.listdir(imgpath)

def changeGray():
    for files in filelist:
        # img = cv2.imread(imgpath+files)
        img = cv2.imread(imgpath+files, cv2.IMREAD_GRAYSCALE)
        # cv2.imwrite(imgpath+"9"+files, img,cv2.CV_16U)
        cv2.imwrite(imgpath+files, img)
        # cv2.imshow("img_gray",img)
        # cv2.waitKey()
def changeName():
    for files in filelist:
        x = files.split("9")
        k = x[1]
        if (len(x) > 2):
            k = k + "9" + x[2]
        img = cv2.imread(imgpath + files, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(imgpath + k, img)
if __name__ == '__main__':
    changeGray()
    # changeName()