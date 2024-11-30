import numpy as np
import cv2
import os
import uuid
import matplotlib.pyplot as plt
# import iris_log
#ImagePath="\image"



def recflection_remove(img):
    ret, mask = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)                          
    dilation = cv2.dilate(mask, kernel, iterations=1)           
    dst = cv2.inpaint(img, dilation, 5, cv2.INPAINT_TELEA)     
    return dst


def processing(image_path,r):                                
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(image, 11)
    ret, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=ret, param2=30, minRadius=20,
                               maxRadius=100)
    circles = circles[0, :, :]  # 提取为二维
    circles = np.int16(np.around(circles))  # 四舍五入，取整
    for i in circles[:]:
        image = image[i[1] - i[2] - r:i[1] + i[2] + r, i[0] - i[2] -r:i[0] + i[2] + r]
        radus = i[2]
    print(image)
    return image, radus


def daugman_normalizaiton(image, height, width, r_in, r_out):       # Daugman归一化，输入为640*480,输出为width*height
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)  # Theta values
    r_out = r_in + r_out
    # Create empty flatten image
    flat = np.zeros((height,width, 3), np.uint8)
    circle_x = int(image.shape[0] / 2)
    circle_y = int(image.shape[1] / 2)

    for i in range(width):
        for j in range(height):
            theta = thetas[i]  # value of theta coordinate
            r_pro = j / height  # value of r coordinate(normalized)

            # get coordinate of boundaries
            Xi = circle_x + r_in * np.cos(theta)
            Yi = circle_y + r_in * np.sin(theta)
            Xo = circle_x + r_out * np.cos(theta)
            Yo = circle_y + r_out * np.sin(theta)

            # the matched cartesian coordinates for the polar coordinates
            Xc = (1 - r_pro) * Xi + r_pro * Xo
            Yc = (1 - r_pro) * Yi + r_pro * Yo

            color = image[int(Xc)][int(Yc)]  # color of the pixel

            flat[j][i] = color
    return flat  # liang

class Images(object):

    def __init__(self,path):         

        self.path = path
        self._r = 55
        self._height = 60
        self._width = 360

    def get_imagepath(self):
        path_images=[]
        file_images=[]
        for filename in os.listdir(self.path):
            file_images.append(filename)
            path_images.append(os.path.join(self.path,filename))
        return file_images,path_images


    @property
    def collect_images(self):         
        success_images = []
        fail_id = []
        success_id = []
        i=0
        for file in self.path:

            if file.ImagePath.endswith(".jpg"):
                image_path = file.ImagePath
                image_id = file.ImageId
                # process_log = iris_log.debug_log()
                #print(image_path)ImagePath
                image_roi, round = processing(image_path,self._r)
                plt.imshow(image_roi )
                plt.show()
                image_roi = recflection_remove(image_roi)
                plt.imshow(image_roi )
                plt.show()
                image_nor =  daugman_normalizaiton(image_roi,self._height, self._width, round, self._r)
                image_nor = cv2.cvtColor(image_nor, cv2.COLOR_BGR2GRAY)
                image_nor = cv2.equalizeHist(image_nor)
                # show the image_nor
                # cv2.imshow("image_nor", image_nor)
                plt.imshow(image_nor, cmap="gray")
                plt.show()
                success_images.append(image_nor)
                success_id.append(image_id)
                print("Success loading:"+" " + "%s"%image_path)
                i = i + 1


        return np.array(success_images,dtype=np.float32).reshape(-1,60,360,1),success_id,fail_id
    



def insert_into_Iris_Reco_Image(path):
    results =[]
    image_path_list = []
    image_list = os.listdir(path)
    for file in image_list:
        image_path  = os.path.join(path,file)
        #if (iris_fuzzydetect.fuzzy_detect(image_path,file)):
        image_path_list.append(image_path)

    class Iris_path (object):
        pass
    for i in range(len(image_path_list)):
        result = Iris_path()
        result.ImageId = uuid.uuid1()
        result.ImagePath = image_path_list[i]
        results.append(result)
    print("Submit success")
    return results

results = insert_into_Iris_Reco_Image("C:\\Users\MASK\\Documents\\GitHub\\Iris_Recognition")
# results = insert_into_Iris_Reco_Image("G:\\4th year biomedical\\Biometrics\\trash\\.s\\Iris_Recognition")
images = Images(results)
images.collect_images()