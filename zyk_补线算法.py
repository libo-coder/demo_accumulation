import cv2
import numpy as np
import math
import json
from functools import cmp_to_key

def cmp_fun(elem1,elem2):
    return elem1['type']-elem2['type']

def data_search(data, level):
    list = []
    temp = []
    for i in range(len(data)):
        if data[i] > level:
            temp.append(data[i])
        else:
            if len(temp) > 0:
                list.append(temp)
            temp = []
    if len(temp) > 0:
        list.append(temp)
    return list

# 补全两条竖线之间距离<=d的区域
# 0 0 0 0 0 255 255 255 255 0 0 0 0 255
# 
# 
#  255 0 255 0 0 0 0 0 0
# 补全 d <= 2
# 0 0 0 0 0 255 255 255 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0
def fill_vline(line, distance):
    arr = np.array(line)
    pos = np.where(arr < 128)
    pos1 = (np.roll(pos[0], -1),)
    if len(pos[0]) == 0:
        return arr
    pos1[0][-1] = pos[0][-1]
    d = pos1[0] - pos[0]
    d_pos = np.where((d > 1) & (d < distance+2))
    for idx, dp in enumerate(pos[0][d_pos]):
        arr[dp+1:pos1[0][d_pos][idx]] = 0
    return arr

'''
    补线算法:
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓    y0
        ┃   ┃      ┃    O0         ┃
        ┣━━━┻━━━━━━━━━━━━━━━━━━━━━━┫    y1
        ┃        ┃      O1 ┃       ┃
        ┣━━━━━━━━━━━━━━━━━━━━━━━━━━┫    y2
        ┃           O2   ┃     ┃   ┃
        ┃      ┃               ┃   ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛    y3

        竖线补线算法:

        两根竖线之间的间隔小于 阈值 d 即补全两竖线之间

        将图像根据 y0,y1,y2.. 划分区域， 设划分区域为O0, O1, O2
        设 O 域中线 h 分别为 y0+(y1-y1)/2 ,y1 + (y2-y1)/2 ,y2+(y3-y2)/2 ...
           O 域起始坐标为 yo1
           O 域终止坐标为 yo2 
        找纵向连通域，长度超过阈值 lth
        设纵向连通域(h1, h2), h1表示起始横坐标， h2 表示终止横坐标
        若 h1 在 O 域中线上方，及h1 > h, 联通 h1 -> O域起始坐标 yo1
        若 h1 在 O 域中线下方，不处理
        若 h2 在 O 域中线下方, 及h2 < h, 联通 h2 -> O域终止坐标 yo2
        若 h2 在 O 域中线上方，不处理
'''
def fill_hline(img):
    _,binImg = cv2.threshold(img, 128, 255,cv2.THRESH_BINARY_INV)
    height, width = binImg.shape[:2]
    HLine = []
    for i in range(height):
        pos = np.where(binImg[i]<128)
        if binImg[i][pos].size > int(width*0.2):
            #print('i:' + str(i) + 'binImg[i][pos].size:' + str(mask[i][pos].size) + '\n')
            binImg[i] = 0
            HLine.append(i)

    key_hline = []
    for idx,i in enumerate(HLine):
        if HLine[idx] - HLine[idx-1] > 30:
            key_hline.append([HLine[idx-1], HLine[idx]])
    #print(key_hline)
    if key_hline == [] or HLine == []:
        return img
    binImg[0:key_hline[0][0],:] = 0
    binImg[key_hline[-1][-1]:,:] = 0
    for i in range(width):
            hl = np.arange(0,height)
            # 连接两条距离接近20像素的竖线
            pos = np.where(binImg[:,i]<128)
            if pos[0].size > int(height*0.7):
                #print('i:' + str(i) + 'binImg[i][pos].size:' + str(mask[i][pos].size) + '\n')
                binImg[:,i] = 0
            if pos[0].size > int(height*0.1):
                pos1 = np.where(binImg[:,i] > 128)
                hl[pos1] = 0
                hl = fill_vline(hl, 20)
                lines = data_search(hl,0)
                for line in lines:
                    # 忽略长度小于 height* 0.08 的短线
                    if len(line) < int(height*0.01):
                        continue
                    # 边缘线补全
                    if i < int(width * 0.01):
                        binImg[:,i] = 0
                        continue
                    elif i > int(width * 0.99):
                        binImg[:,i] = 0
                        continue
                    # 中间竖线按补线算法计算
                    for key in key_hline:
                        if line[0] < (key[0] + (key[1] - key[0])*3/5):
                        #if line[0] < ((key[0] + key[1])*2/3):
                            binImg[key[0]:line[0],i] = 0
                        if line[-1] > ((key[0] + (key[1] - key[0])*1/5)):
                            binImg[line[-1]:key[1],i] = 0
                    #print("line[0]:" + str(line[0]) + "line[-1]:" + str(line[-1]))
            binImg[:,i] = fill_vline(binImg[:,i],25)
    _,binImg = cv2.threshold(binImg, 128, 255,cv2.THRESH_BINARY_INV)
    return binImg

def get_roi_type(img_height, img_width, x,y):
    #img_height, img_width = img.shape[:2]
    scale_x = x / img_width
    scale_y = y / img_height
    if scale_y < 0.21:
        if scale_x < 0.043:
            return 0
        elif scale_x < 0.5734: 
            return 1
        elif scale_x < 0.65:
            return 2
        else:
            return 3
    elif scale_y < 0.65:
        if scale_x < 0.2688:
            return 4
        elif scale_x < 0.387:
            return 5
        elif scale_x < 0.45:
            return 6
        elif scale_x < 0.55:
            return 7
        elif scale_x < 0.65:
            return 8
        elif scale_x < 0.79926:
            return 9
        elif scale_x < 0.892:
            return 10
        else :
            return 11
    elif scale_y < 0.75:
        if scale_x < 0.26229:
            return 12
        else:
            return 13
    else:
        if scale_x < 0.04254:
            return 14
        elif scale_x < 0.57:
            return 15
        elif scale_x < 0.662:
            return 16
        else :
            return 17

def get_roi_by_type(rois, type_id):
    for roi in rois:
        if roi["type"] == type_id:
            return roi["img"]
        
    return np.array([])
standard_width = 1160
standard_height = 750
def RotationImg(img):
    #cv2.imshow('origin', img)
    img = cv2.resize(img,(table_width,table_height))
    kernel_size = 5
    #blur_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    low_threshold = 50
    high_threshold = 200
    edges = cv2.Canny(img, low_threshold, high_threshold)
    #cv2.imshow('edges',edges)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 360  # angular resolution in radians of the Hough grid
    threshold = 300  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 250  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    angle = 0.0
    for line in lines:
        for x1,y1,x2,y2 in line:
            if abs(math.atan2(y2 - y1, x2 - x1)) > math.pi/4:
                continue
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),1)
            t = math.atan2(y2 - y1, x2 - x1)
            angle += 180 / math.pi * t
    angle = angle / len(lines)
    
    roi = img.copy()
    #cv2.imshow('roi', roi)

    rows,cols = img.shape[:2]
    m2 = cv2.getRotationMatrix2D((cols/2, rows/2), angle,1)
    ratationedImg = cv2.warpAffine(roi,m2,(roi.shape[1], roi.shape[0]),borderMode=cv2.BORDER_REFLECT,borderValue=(255,255,255))
    return ratationedImg
def GetContoursPic(img):
    #cv2.imshow('origin', img)
    img = cv2.resize(img,(table_width,table_height))
    kernel_size = 5
    blur_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    low_threshold = 50
    high_threshold = 200
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    #cv2.imshow('edges',edges)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 360  # angular resolution in radians of the Hough grid
    threshold = 300  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 250  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    angle = 0.0
    for line in lines:
        for x1,y1,x2,y2 in line:
            if abs(math.atan2(y2 - y1, x2 - x1)) > math.pi/4:
                continue
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),1)
            t = math.atan2(y2 - y1, x2 - x1)
            angle += 180 / math.pi * t
    angle = angle / len(lines)
    
    roi = img.copy()
    #cv2.imshow('roi', roi)
    rows,cols = img.shape[:2]
    m2 = cv2.getRotationMatrix2D((cols/2, rows/2), angle,1)
    ratationedImg = cv2.warpAffine(roi,m2,(roi.shape[1], roi.shape[0]),borderMode=cv2.BORDER_REFLECT,borderValue=(255,255,255))
    #cv2.imshow('rotation', ratationedImg)
    raw = ratationedImg
    
    secondFindImg = cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
    ret,gray = cv2.threshold(secondFindImg, 80, 200, cv2.THRESH_BINARY)
    contours2,hier = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    dst = np.zeros_like(raw)
    for contour in contours2:
        rect = cv2.boundingRect(contour)
        x1,y1,w,h = rect[:4]
        if  w*h < 600:
            continue
        dst = raw[y1:y1+h, x1:x1+w]

        #cv2.imshow('dst',  dst)
        #cv2.imwrite('image/rois/dst.jpg', dst)
    
    return dst


#table_width = 985
#table_height = 565
table_width = 1395
table_height = 800
def getTableRoi(img):
    #rsz = cv2.resize(img,(table_width,table_height))
    rsz = img
    #gray = cv2.cvtColor(rsz,cv2.COLOR_BGR2GRAY)
    b,g,r = cv2.split(rsz)
    gray = r
    #cv2.imshow('gray',gray)
    gray_not = cv2.bitwise_not(gray)
    #自适应阈值
    binImg = cv2.adaptiveThreshold(gray_not,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,19,-2)
    #cv2.imshow('binary',binImg)

    horizontal = binImg.copy()
    vertical = binImg.copy()

    rows,cols =horizontal.shape[:2]
    h_scale = 55
    horizontalsize = int(cols / h_scale)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(horizontalsize,1))

    horizontal = cv2.erode(horizontal, horizontalStructure,anchor=(-1,-1))
    horizontal = cv2.dilate(horizontal,horizontalStructure,anchor=(-1,-1))
    #cv2.imshow("horizontal", horizontal)

    v_scale = 18
    verticalsize = int(rows / v_scale)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(1,verticalsize))
    vertical = cv2.erode(vertical, verticalStructure,anchor=(-1,-1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1,-1))
    #cv2.imshow('vertical', vertical)
    mask = horizontal + vertical
    #cv2.imshow('mask', mask)

    joints = cv2.bitwise_and(horizontal,vertical)

    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,offset=(0,0))
    joints_contours,joints_hierarchy = cv2.findContours(joints, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    points = []
    for contour in joints_contours:
        for point in contour:
            points.append(point[0])
    boundRect = cv2.boundingRect(np.array(points))
    x,y,w,h = boundRect[:4]
    #cv2.rectangle(joints, (x,y), (x+w,y+h),(255,255,255),1,8,0)
    #cv2.imshow('joints', joints)

    rois = []
    maxArea_roi = {"area":0, "roi":np.zeros_like(img)}
    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 100:
            continue
        contour_ploy = cv2.approxPolyDP(contour,3,True)
        boundRect = cv2.boundingRect(contour_ploy)
        x,y,w,h = boundRect[:4]
        roi = joints[y:y+h,x:x+w]
        cv2.rectangle(joints, (x,y),(x+w,y+h),(255,255,255),1,8,0)
        # 过滤掉包含交点小于 4 个的 roi 区域
        joints_contours,joints_hierarchy = cv2.findContours(roi,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        if len(joints_contours) <= 4:
            continue
        rois.append(rsz[y:y+h,x-1:x+w].copy())
        # 取面积最大的 roi 区域
        if cv2.contourArea(contour) > maxArea_roi["area"]:
            #print(" contours :" + str(cv2.contourArea(contour)))
            maxArea_roi["area"] = cv2.contourArea(contour)
            maxArea_roi["roi"] = rois[-1]
        #cv2.rectangle(rsz, (x,y),(x+w,y+h),(0,255,0),1,8,0)

    '''
    for roi in rois:
        cv2.imshow("roi", roi)
        #cv2.waitKey()
    
    cv2.imshow('contours', rsz)
    cv2.waitKey()
    '''
    
    if len(rois) == 0:
        return rsz
    return maxArea_roi["roi"]

def SplitRect(img):
    rect_rois = []
    # x：区域中心点 x 坐标 
    # y: 区域中心点 y 坐标
    rect_item = {
        'x' : 0,
        'y' : 0,
        'img': img
    }
    rsz = cv2.resize(img,(table_width,table_height))
    #cv2.imwrite("./image/rois/rsz.jpg", rsz)
    b,g,r = cv2.split(rsz)
    #_,gray = cv2.threshold(r, 220, 255, cv2.THRESH_BINARY)
    gray = r
    #gray = cv2.cvtColor(rsz,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray',gray)
    gray_not = cv2.bitwise_not(gray)
    #cv2.imshow('gray_not',gray_not)
    #自适应阈值
    binImg = cv2.adaptiveThreshold(gray_not,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,19,-2)
    height, width = binImg.shape[:2]
    #binImg[:3] = 255
    #binImg[-3:] = 255
    #binImg[:,:3] = 255
    #binImg[:,-3:] = 255
    '''
    for i in range(width):
        pos = np.where(binImg[:, i-1:i+2] > 128)
        if binImg[:][pos].size > int(height*1.8):
            binImg[:,i-1:i+1] = 255
    '''
    #cv2.imshow('binary',binImg)

    horizontal = binImg.copy()
    vertical = binImg.copy()

    rows,cols =horizontal.shape[:2]
    h_scale = 55
    horizontalsize = int(cols / h_scale)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(horizontalsize,2))

    horizontal = cv2.erode(horizontal, horizontalStructure,anchor=(-1,-1))
    horizontal = cv2.dilate(horizontal,horizontalStructure,anchor=(-1,-1))
    #cv2.imshow("horizontal", horizontal)

    v_scale = 25
    verticalsize = int(rows / v_scale)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(1,verticalsize))
    vertical = cv2.erode(vertical, verticalStructure,anchor=(-1,-1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1,-1))
    #cv2.imshow('vertical', vertical)
    mask = horizontal + vertical
    cv2.imwrite('mask.jpg', mask)
    '''
    for i in range(height):
        pos = np.where(mask[i]>128)
        if mask[i][pos].size > int(width*0.2):
            #print('i:' + str(i) + 'binImg[i][pos].size:' + str(mask[i][pos].size) + '\n')
            mask[i] = 255
    '''
    '''
    for i in range(width):
        pos = np.where(mask[:, i-1:i+2] > 128)
        if binImg[:][pos].size > int(height*0.15):
    '''
            
    #cv2.imshow('mask', mask)
    mask = RotationImg(mask)
    mask = fill_hline(mask)
    cv2.imshow('fill_line_mask', mask)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2000 or area > (img.shape[0] * img.shape[1] / 2):
            continue
        contour_ploy = cv2.approxPolyDP(contour,3,True)
        boundRect = cv2.boundingRect(contour_ploy)
        x,y,w,h = boundRect[:4]
        roi = rsz[y:y+h,x:x+w]
        roi_type = get_roi_type(table_height,table_width,x+w/2,y+h/2)
        #合并相同类型区域
        output = np.zeros_like(roi)
        concateFlag = False
        for rect_roi in rect_rois:
                if rect_roi['type'] == roi_type:
                    #print("x:" + str(rect_roi["x"]) + "cur x: " + str(x + w/2))
                    #print("y:" + str(rect_roi["y"]) + "cur y: " + str(y + h/2))
                    cv2.imshow("roi", roi)
                    cv2.imshow("rect_roi", rect_roi["img"])
                    cv2.waitKey()
                    rect_roi_h,rect_roi_w = rect_roi["img"].shape[:2]
                    rect_roi["x"] = (rect_roi["x"] + (x + w/2)) / 2
                    heigh = np.max([rect_roi_h, h])
                    roi = cv2.resize(roi, (w,heigh))
                    rect_roi["img"] = cv2.resize(rect_roi["img"], (rect_roi_w,heigh))
                    output = np.zeros((heigh,width,3))
                    if rect_roi["x"] > (x + w/2):
                        output = np.concatenate((roi,rect_roi["img"]),axis=1)
                    else :
                        output = np.concatenate((rect_roi["img"],roi),axis=1)
                    rect_roi["img"] = output
                    concateFlag = True
                    break
        if concateFlag == False:
            rect_item = {
                'x' : x + w/2,
                'y' : y + h/2,
                'img': roi,
                'type':get_roi_type(table_height,table_width,x+w/2,y+h/2)
            }
            rect_rois.append(rect_item)
        #cv2.imshow("roi", roi)
        #cv2.imwrite("image/rois/roi_"+ str(i) +".jpg", roi)
        #i += 1


    #joints = cv2.bitwise_and(horizontal,vertical)
    #cv2.imshow('joints', joints)
    return rect_rois

class InvoiceSplit():
    def invoiceSplit(img):
        dstimg = GetContoursPic(img)
        table = getTableRoi(dstimg)
        rect_rois = SplitRect(table)
        return rect_rois