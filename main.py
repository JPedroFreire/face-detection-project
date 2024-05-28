import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(".\sample_imgs\sample7.png", cv.IMREAD_COLOR)

img_luma = img[:, :, 0] * 0.0721  + img[:, :, 1] * 0.7154 + img[:, :, 2] * 0.2125
average_luma = np.mean(img_luma)/255

def light_correction(img, img_luma):
    img = img.astype(np.float32)

    img = img / 255

    gamma = 0.45 #valor de gamma padrão segundo Szeliski para correção
    
    img = img**(1/gamma)
    
    num_top_pixels = int(img_luma.size*0.05)

    # ordenar array
    sorted_indices = np.argsort(img_luma.ravel())

    # selecionar indices com maior valor de luma
    top_indices = sorted_indices[-num_top_pixels:]

    # converter indices para coordenadas
    top_coords = np.column_stack(np.unravel_index(top_indices, img_luma.shape))

    top_pixels = img[top_coords[:, 0], top_coords[:, 1]]

    copy_top_pixels = top_pixels

    copy_average = np.mean(copy_top_pixels)

    correction = 1.0/copy_average

    copy_top_pixels[:, 0] *= correction
    copy_top_pixels[:, 1] *= correction
    copy_top_pixels[:, 2] *= correction

    img = (img*255).astype(np.uint8)
    img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    return img

def gray_world_algorithm(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

    avg_cb = np.average(img[:,:,2])
    avg_cr = np.average(img[:,:,1])
    img[:,:,2] = img[:,:,2] - ((avg_cb - 128) * (img[:,:,0] / 255.0) * 2.2)
    img[:,:,1] = img[:,:,1] - ((avg_cr - 128) * (img[:,:,0] / 255.0) * 2.2)
    
    return img

if(average_luma >= 0.7):
    img = gray_world_algorithm(img)
else:
    img = light_correction(img, img_luma)

#-------------------------------------------------

#-------------- Color Space transformation -------
Wcb = 46.97
WLcb = 23
WHcb = 14
Wcr = 38.76
WLcr = 20
WHcr = 10
Kl = 125
Kh = 188
Ymin = 16
Ymax = 235

def center_cr_y(Y):
    if(Y < Kl):
        result = 154 - ((Kl - Y)*(154 - 144))/(Kl - Ymin)
    elif(Y > Kh):
        result = 154 + ((Y - Kh)*(154 - 132))/(Ymax - Kh)
    
    return result

def center_cb_y(Y):
    result = 222
    if(Y < Kl):
        result = 108 + ((Kl - Y)*(118 - 108))/(Kl - Ymin)
    elif(Y > Kh):
        result = 108 + ((Y - Kh)*(118 - 108))/(Ymax - Kh)
    
    return result

def wcr_y(Y):
    if(Y < Kl):
        result = WLcr + ((Y - Ymin)*(Wcr - WLcr))/(Kl - Ymin)
    elif(Y > Kh):
        result = WHcr + ((Ymax - Y)*(Wcr - WHcr))/(Ymax - Kh)
    
    return result

def wcb_y(Y):
    if(Y < Kl):
        result = WLcb + ((Y - Ymin)*(Wcb - WLcb))/(Kl - Ymin)
    elif(Y > Kh):
        result = WHcb + ((Ymax - Y)*(Wcb - WHcb))/(Ymax - Kh)
    
    return result

def cr_y_transform(pixel):
    y = pixel[0]
    if(Kl <= y <= Kh):
        pass
    else:
        pixel[1] = (pixel[1] - center_cr_y(y)) *  (Wcr/wcr_y(y)) #+ center_cr_y(y)#+ center_cr_y(Kh) OU + 

def cb_y_transform(pixel):
    y = pixel[0]
    if(Kl <= y <= Kh):
        pass
    else:
        pixel[2] = (pixel[2] - center_cb_y(y)) * (Wcb/wcb_y(y)) #+ center_cb_y(y)#+ center_cb_y(Kh) OU 

#np.apply_along_axis(cb_y_transform,axis=2,arr=img)
#np.apply_along_axis(cr_y_transform,axis=2,arr=img)

#-------------------------------------------------

#-------------- Elliptical Skin Model ------------
theta = 2.53
sin_theta = math.sin(theta)
cos_theta = math.cos(theta)
mat = [[cos_theta, sin_theta],[-sin_theta, cos_theta]]
cx = 109.38
cy = 152.02
ecx = 1.6
ecy = 2.41
a = 25.39
b = 14.03

col = []

def mat_mult(pixel):
  pixel_to_mult = [pixel[2]-cx, pixel[1]-cy]
  if(ellipse_bound(np.matmul(mat,pixel_to_mult))):
    col.append('red')
  else:
    col.append('blue')
 
def ellipse_bound(point):
  x = point[0]
  y = point[1]
  return ((x-ecx)**2)/(a**2) + ((y-ecy)**2)/(b**2) <= 1

def ellipse(pixel):
  pixel_to_mult = [pixel[2]-cx, pixel[1]-cy]
  x,y = np.matmul(mat,pixel_to_mult)
  return ((x-ecx)**2)/(a**2) + ((y-ecy)**2)/(b**2) <= 1

mask = np.apply_along_axis(ellipse, axis=2, arr=img)

mask = mask.astype(np.uint8) * 255
#-------------------------------------------------

#-------------- Eye Maps -------------------------

def eye_mapC(pixel):
    cb2 = (pixel[2]**2)/255
    n_cr2 = ((255 - pixel[1])**2)/255
    cbcr_div = min(255, max(0, round((pixel[2]/pixel[1]) * 255))) # ValueError: cannot convert float NaN to integer
    return ((cb2 + n_cr2 + (cbcr_div))/3)

eyemapC = np.apply_along_axis(eye_mapC, axis=2, arr=img).astype(np.uint8)
eyemapC = cv.equalizeHist(eyemapC)

se = np.ones((3, 3), np.uint8) #structuring element, descrito no artigo Scale-Space Properties of the Multiscale Morphological Dilation-Erosion

img_eroded = cv.erode(img[:,:,0], se) + 1
img_dilated = cv.dilate(img[:,:,0], se) 

eyemapL = np.divide(img_dilated, img_eroded).astype(np.uint8)

eyemap = cv.multiply(eyemapL, eyemapC).astype(np.uint8)
eye_dilated = cv.dilate(eyemap, se)

closed_eye_map = cv.morphologyEx(eye_dilated, cv.MORPH_CLOSE, se)

#-------------------------------------------------

#-------------- Mouth Map ------------------------
'''
#somatórios descritos no artigo (média de cr^2 divido por médias de cr/cb da máscara)
eta = 0.95 - (1)

def mouth_mask(pixel):
    cr2 = (pixel[1]**2)/255
    crcb_div = min(255, max(0, round((pixel[1]/pixel[2]) * 255)))
    return cr2 * (cr2 - eta * crcb_div)**2
'''
#-------------------------------------------------

#-------------- Face Boundary and Score --------------------

#hough transform
#calculate face score based on masks and face boundary

#-------------------------------------------------

img = cv.cvtColor(img, cv.COLOR_YCrCb2BGR)

cv.imshow("img", img)
cv.imshow("mask", mask)
cv.imshow("eye map", closed_eye_map)

cv.waitKey(0)
cv.destroyAllWindows()