#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
import pytesseract
from bs4 import BeautifulSoup
import re


# In[18]:


#图像自适应二值化
def img_adpt_thre(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转换了灰度化 
    img_thre = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
    return img_thre


# In[34]:

#均值二值化
def img_thre(fn ='',img = None):
    if not isinstance(img,np.ndarray):
        img = cv2.imread(fn)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y, x = gray.shape[:2]
    m = np.reshape(gray, (1, x * y))
    mean = m.sum() / (x * y)
    if mean > 128:
        img_thre = img_adpt_thre(img)
    else:
        retval, img_thre = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
        img_thre = convert_color(img_thre)
    return img_thre


# #### 统计像素
# 返回4个list，每一行的白像素，每一行的黑像素，每一列的白像素，每一列的黑像素。

# In[35]:


#统计行和列的黑白像素
def count_r_c_pix(img):
    #预生成统计行列黑白像素的数组
    height,width = img.shape[:2]
    rw = [0]*height
    rb = [0]*height
    cw = [0]*width
    cb = [0]*width
    # 计算每一列的黑白色像素总和
    for i in range(width):
        for j in range(height):
            if img[j][i] == 255:
                cw[i] += 1
                rw[j] += 1
            if img[j][i] == 0:
                cb[i] += 1
                rb[j] += 1
    return rw,rb,cw,cb


# In[36]:


def normalizing_img(img):
    height, width = img.shape[:2]
    w, b = 0, 0
    for i in range(width):
        for j in range(height):
            if img[j][i] == 255:
                w += 1
            elif img[j][i] == 0:
                b += 1
    if w < b:
        for i in range(width):
            for j in range(height):
                img[j][i] = 255 - img[j][i]
    return img


def convert_color(img):
    height, width = img.shape[:2]
    for i in range(width):
        for j in range(height):
            img[j][i] = 255 - img[j][i]
    return img


# In[37]:


def image_to_pdf_or_hocr(fn ='',img = None,lang = 'eng',extension='hocr'):
    img = img_thre(fn,img)
    content = pytesseract.image_to_pdf_or_hocr(img,lang=lang,extension=extension)
    return content


# In[ ]:




