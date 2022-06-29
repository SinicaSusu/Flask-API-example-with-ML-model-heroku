# -*- coding: UTF-8 -*-
from app import app
import os
import cv2
import math
import json
import numpy as np
from openpyxl import load_workbook
from openpyxl import Workbook
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow, show
from flask import Flask, Response, request, Request
from flask_cors import CORS


@app.route('/')
def index():
    return 'Flask API started'

#圖片去背路徑
@app.route('/img_calculate', methods=["POST"])
def calculate_img():
    # 接收前端传来的图片  image定义上传图片的key
    upload_img = request.files['image']
    print(type(upload_img), upload_img)
    # <class 'werkzeug.datastructures.FileStorage'> <FileStorage: 'phone.jpg' ('image/jpeg')>

    # 获取到图片的名字
    img_name = upload_img.filename

    # 把前端上传的图片保存到后端
    upload_img.save(os.path.join('static/image/', upload_img.filename))

    # 对后端保存的图片进行镜像处理
    img_path = os.path.join('static/image/', upload_img.filename)
    print('path', img_path)
        
    bgr_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    # 高斯模糊
    blur_img = cv2.GaussianBlur(rgb_img,(99, 99), 0)
    # 轉灰階
    gray_img = cv2.cvtColor(blur_img, cv2.COLOR_RGB2GRAY)
    #二值化
    ret, threshold = cv2.threshold(gray_img, 111, 255, cv2.THRESH_BINARY_INV)
        
    #尋找口香糖的輪廓
    all_contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = list(all_contours)
    area_list = list(map(cv2.contourArea, all_contours))
    gum_contours = []
    post1 = area_list.index(max(area_list))
    gum_contours.append(all_contours[post1])
    del all_contours[post1]
    del area_list[post1]
    post2 = area_list.index(max(area_list))
    gum_contours.append(all_contours[post2])

    #建立遮罩
    cimg = np.zeros_like(rgb_img)
    cimg[:, :, :] = 255
    mask = cv2.drawContours(cimg, gum_contours, -1, color = (0, 0, 0), thickness = -1)

    #利用遮罩與原圖做計算
    cropped_img = cv2.bitwise_or(rgb_img, mask)

    #縮小成512*512
    resized_img = cv2.resize(cropped_img, (512, 512), interpolation = cv2.INTER_AREA)

    #將3D轉為2D
    img_color = resized_img.reshape((resized_img.shape[0] * resized_img.shape[1], 3))

    #挑出遮罩內的元素
    pixel_in_mask = []
    for i in range(0, 262144):
        a = img_color[i]
        if ((a == [255, 255, 255]).all()) == False:
            pixel_in_mask.append(a)
    colors_H = []
    colors_S = []
    colors_I = []
    for i in range(0, len(pixel_in_mask)):
        r = pixel_in_mask[i][0] / 255
        g = pixel_in_mask[i][1] / 255
        b = pixel_in_mask[i][2] / 255
        
        Sum = r + g + b
        Min = min(r, g, b)
        
        num = 0.5 * ((r - g) + (r - b))
        den = np.sqrt((r - g) **2 + (r - b) * (g - b))
        theta = float(np.arccos(num / den))
        
        #計算H(Hue)
        if den == 0:
            H = 0
        elif g >= b:
            H = (theta * 180) / np.pi
        elif g < b:
            H = ((2 * np.pi - theta) * 180) / np.pi
        colors_H.append(H)
        
        
        #計算S(Saturation)
        if Sum == 0:
            S = 0
        else:
            S = 1 - ((3 * Min) / Sum)
        colors_S.append(S)
        
        #計算I(Intensity)
        I = Sum / 3.0
        colors_I.append(I)

    #計算H
    nh = len(colors_H)
        
    #取所有像素H的值
    hue = []
    for i in colors_H:
        hu = float(i)
        hue.append(hu)
        i += 1

    #取H總和
    sumhue = sum(hue)

    #平均
    H = sumhue / nh

    #計算HSD
    nhsd = len(colors_H)

    #取H的sin值與cos值
    cos = []
    sin = []
    for i in colors_H:
        c = math.cos(math.radians(i))
        s = math.sin(math.radians(i))
        cos.append(c)
        sin.append(s)
        i += 1

    #取H的sin值總和的平方值與cos值總和的平方值
    sqofsumcos = math.pow(sum(cos), 2)
    sqofsumsin = math.pow(sum(sin), 2)

    #總和
    z = sqofsumcos + sqofsumsin


    #計算變異數
    VOH = 1 - math.pow(z, .5) / nhsd

    #計算標準差
    SDOH = math.pow(VOH, .5)

    #計算S
    ns = len(colors_S)
        
    #取所有像素S的值
    saturation = []
    for i in colors_S:
        sat = float(i)
        saturation.append(sat)
        i += 1

    #取S總和
    sumsat = sum(saturation)

    #平均
    S = sumsat / ns

    #計算SSD
    nssd = len(colors_S)
        
    #取所有像素S的值
    saturation = []
    for i in colors_S:
        sat = float(i)
        saturation.append(sat)
        i += 1


    #計算變異數
    VOS = np.var(saturation)

    #計算標準差
    SDOS = np.std(saturation)

    #計算I
    ni = len(colors_I)
        
    #取所有像素I的值
    intensity = []
    for i in colors_I:
        inten = float(i)
        intensity.append(inten)
        i += 1

    #取S總和
    sumint = sum(intensity)

    #平均
    I = sumint / ni

    #計算ISD
    nisd = len(colors_I)
        
    #取所有像素I的值
    intensity = []
    for i in colors_I:
        inten = float(i)
        intensity.append(inten)
        i += 1


    #計算變異數
    VOI = np.var(intensity)

    #計算標準差
    SDOI = np.std(intensity)
    
    cv2.imwrite(os.path.join('static/image/', 'res_' + upload_img.filename), resized_img)
    print(H, SDOH, S, SDOS, I, SDOI)

    #數值存成dict
    python_dict = {'H':H,'HSD':SDOH,'S':S,'SSD':SDOS,'I':I,'ISD':SDOI}
    print(python_dict, type(python_dict))
    j = json.dumps(python_dict, indent = 4)
    print(j, type(j))

    # 把图片读成二进制，返回到前端
    #image = open(os.path.join('static/image/', 'res_' + upload_img.filename), mode='rb')
    j = Response(j,content_type="application/x-www-form-urlencoded")
    return j

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=False)
