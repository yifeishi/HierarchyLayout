import random
import numpy as np
import os
import util
import math

def getPTS(pts_file,colorPara=1):
    fpts = open(pts_file)
    count = 0
    while 1:
        line = fpts.readline()
        if not line:
            break
        count = count + 1
    if count==0:
        return np.zeros(6)
    points = np.zeros((count,6))
    count = 0
    fpts = open(pts_file)
    while 1:
        line = fpts.readline()
        if not line:
            break
        L = line.split(' ')
        points[count,0] = float(L[0])
        points[count,1] = float(L[1])
        points[count,2] = float(L[2])
        points[count,3] = float(L[3])/colorPara
        points[count,4] = float(L[4])/colorPara
        points[count,5] = float(L[5])/colorPara
        count = count + 1
    return points

# xMax,yMax,zMax,xMin,yMin,zMin to size,cen
def AabbFeatureTransformer(obj_obb_fea_tmp):
    obj_obb_fea = np.zeros(6)
    obj_obb_fea[0] = obj_obb_fea_tmp[0] - obj_obb_fea_tmp[3]
    obj_obb_fea[1] = obj_obb_fea_tmp[1] - obj_obb_fea_tmp[4]
    obj_obb_fea[2] = obj_obb_fea_tmp[2] - obj_obb_fea_tmp[5]
    obj_obb_fea[3] = (obj_obb_fea_tmp[0] + obj_obb_fea_tmp[3])*0.5
    obj_obb_fea[4] = (obj_obb_fea_tmp[1] + obj_obb_fea_tmp[4])*0.5
    obj_obb_fea[5] = (obj_obb_fea_tmp[2] + obj_obb_fea_tmp[5])*0.5
    return obj_obb_fea

# size,cen to xMax,yMax,zMax,xMin,yMin,zMin
def AabbFeatureTransformerReverse(obj_obb_fea_tmp):
    obj_obb_fea = np.zeros(6)
    obj_obb_fea[0] = obj_obb_fea_tmp[3] + obj_obb_fea_tmp[0]*0.5
    obj_obb_fea[1] = obj_obb_fea_tmp[4] + obj_obb_fea_tmp[1]*0.5
    obj_obb_fea[2] = obj_obb_fea_tmp[5] + obj_obb_fea_tmp[2]*0.5
    obj_obb_fea[3] = obj_obb_fea_tmp[3] - obj_obb_fea_tmp[0]*0.5
    obj_obb_fea[4] = obj_obb_fea_tmp[4] - obj_obb_fea_tmp[1]*0.5
    obj_obb_fea[5] = obj_obb_fea_tmp[5] - obj_obb_fea_tmp[2]*0.5
    return obj_obb_fea

def computeAabbIou(boxA,boxB): # size,center
    boxA = AabbFeatureTransformerReverse(boxA)
    boxB = AabbFeatureTransformerReverse(boxB)
    xA = max(boxA[3], boxB[3])
    yA = max(boxA[4], boxB[4])
    zA = max(boxA[5], boxB[5])
    xB = min(boxA[0], boxB[0])
    yB = min(boxA[1], boxB[1])
    zB = min(boxA[2], boxB[2])

    if xA - xB > 0 or yA - yB > 0 or zA - zB > 0:
        interArea = 0
        return 0
    else:
        interArea = (xB - xA) * (yB - yA) * (zB - zA)

    boxAArea = (boxA[0] - boxA[3]) * (boxA[1] - boxA[4]) * (boxA[2] - boxA[5])
    boxBArea = (boxB[0] - boxB[3]) * (boxB[1] - boxB[4]) * (boxB[2] - boxB[5])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def PTS2AABB(points):
    aabb = np.zeros(6)
    aabb[0] = max(points[:,0])
    aabb[1] = max(points[:,1])
    aabb[2] = max(points[:,2])
    aabb[3] = min(points[:,0])
    aabb[4] = min(points[:,1])
    aabb[5] = min(points[:,2])
    aabb = ObbFeatureTransformerReverse(aabb)
    return aabb

def PTS2OBB(points):
    mu = points[:,0:2].mean(axis=0)
    points[:,0:2] = points[:,0:2] - mu
    hull_points = qhull2D(points[:,0:2])
    hull_points = hull_points[::-1]
    (rot_angle, area, width, height, center_point, corner_points) = minBoundingRect(hull_points)

    eigenvectors = np.zeros((2,2))
    eigenvectors[0,0] = math.cos(rot_angle)
    eigenvectors[0,1] = math.sin(rot_angle)
    eigenvectors[1,0] = math.sin(rot_angle)
    eigenvectors[1,1] = -math.cos(rot_angle)

    points[:,0:2] = points[:,0:2] + mu
    obb = np.zeros(8)
    obb[0] = (max(points[:,0])+min(points[:,0]))*0.5
    obb[1] = (max(points[:,1])+min(points[:,1]))*0.5
    obb[2] = (max(points[:,2])+min(points[:,2]))*0.5
    
    projectionMax = -9999
    projectionMin = 9999
    for i in range(0,points.shape[0]):
        projection = (points[i,0] - obb[0]) * eigenvectors[0,0] + (points[i,1] - obb[1]) * eigenvectors[0,1]
        if projectionMax < projection:
            projectionMax = projection
        elif projectionMin > projection:
            projectionMin = projection
    len1 = projectionMax - projectionMin
    offset1 = (projectionMax + projectionMin)*0.5

    projectionMax = -9999
    projectionMin = 9999
    for i in range(0,points.shape[0]):
        projection = (points[i,0] - obb[0]) * eigenvectors[1,0] + (points[i,1] - obb[1]) * eigenvectors[1,1]
        if projectionMax < projection:
            projectionMax = projection
        elif projectionMin > projection:
            projectionMin = projection
    len2 = projectionMax - projectionMin
    offset2 = (projectionMax + projectionMin)*0.5
    
    obb[0] += offset1 * eigenvectors[0,0] + offset2 * eigenvectors[1,0]
    obb[1] += offset1 * eigenvectors[0,1] + offset2 * eigenvectors[1,1]
    
    if len1 > len2:
        obb[3] = len1
        obb[4] = len2
        obb[6] = eigenvectors[0,0]
        obb[7] = eigenvectors[0,1]
    else:
        obb[3] = len2
        obb[4] = len1
        obb[6] = eigenvectors[1,0]
        obb[7] = eigenvectors[1,1]
    
    obb[5] = max(points[:,2]-min(points[:,2]))
    obb_new = np.zeros(8)
    obb_new[0:3] = obb[3:6]
    obb_new[3:6] = obb[0:3]
    obb_new[6:8] = obb[6:8]
    return obb_new

def obb2Aabb(obb):
    aabb = np.zeros(6)
    if abs(obb[6]) > 0.9:
        aabb[0],aabb[1] = obb[0],obb[1]
    elif abs(obb[7]) > 0.9:
        aabb[0],aabb[1] = obb[1],obb[0]
    else:
        aabb[0] = aabb[1] = max(obb[1],obb[0])
    aabb[2] = obb[2]
    aabb[3:6] = obb[3:6]
    return aabb

def getObbVertices(obb):
    points = np.zeros((8,3))
    cen = np.zeros(3)
    leng = np.zeros(3)
    ax1 = np.zeros(3)
    ax2 = np.zeros(3)
    ax3 = np.zeros(3)
    cen[0] = float(obb[3])
    cen[1] = float(obb[4])
    cen[2] = float(obb[5])
    leng[0] = float(obb[0]) * 0.5
    leng[1] = float(obb[1]) * 0.5
    leng[2] = float(obb[2]) * 0.5
    ax1[0] = float(obb[6])
    ax1[1] = float(obb[7])
    ax1[2] = float(0)
    ax3[0] = float(0)
    ax3[1] = float(0)
    ax3[2] = float(1)
    ax2 = np.cross(ax1,ax3)

    offset = np.zeros((8,3))
    offset[0,0] = ax1[0] * leng[0] + ax2[0] * leng[1]  + ax3[0] * leng[2]
    offset[0,1] = ax1[1] * leng[0] + ax2[1] * leng[1]  + ax3[1] * leng[2]
    offset[0,2] = ax1[2] * leng[0] + ax2[2] * leng[1]  + ax3[2] * leng[2]
    offset[1,0] = ax1[0] * leng[0] - ax2[0] * leng[1]  + ax3[0] * leng[2]
    offset[1,1] = ax1[1] * leng[0] - ax2[1] * leng[1]  + ax3[1] * leng[2]
    offset[1,2] = ax1[2] * leng[0] - ax2[2] * leng[1]  + ax3[2] * leng[2]
    offset[2,0] = - ax1[0] * leng[0] - ax2[0] * leng[1]  + ax3[0] * leng[2]
    offset[2,1] = - ax1[1] * leng[0] - ax2[1] * leng[1]  + ax3[1] * leng[2]
    offset[2,2] = - ax1[2] * leng[0] - ax2[2] * leng[1]  + ax3[2] * leng[2]
    offset[3,0] = - ax1[0] * leng[0] + ax2[0] * leng[1]  + ax3[0] * leng[2]
    offset[3,1] = - ax1[1] * leng[0] + ax2[1] * leng[1]  + ax3[1] * leng[2]
    offset[3,2] = - ax1[2] * leng[0] + ax2[2] * leng[1]  + ax3[2] * leng[2]
    offset[4,0] = ax1[0] * leng[0] + ax2[0] * leng[1]  - ax3[0] * leng[2]
    offset[4,1] = ax1[1] * leng[0] + ax2[1] * leng[1]  - ax3[1] * leng[2]
    offset[4,2] = ax1[2] * leng[0] + ax2[2] * leng[1]  - ax3[2] * leng[2]
    offset[5,0] = ax1[0] * leng[0] - ax2[0] * leng[1]  - ax3[0] * leng[2]
    offset[5,1] = ax1[1] * leng[0] - ax2[1] * leng[1]  - ax3[1] * leng[2]
    offset[5,2] = ax1[2] * leng[0] - ax2[2] * leng[1]  - ax3[2] * leng[2]
    offset[6,0] = - ax1[0] * leng[0] - ax2[0] * leng[1]  - ax3[0] * leng[2]
    offset[6,1] = - ax1[1] * leng[0] - ax2[1] * leng[1]  - ax3[1] * leng[2]
    offset[6,2] = - ax1[2] * leng[0] - ax2[2] * leng[1]  - ax3[2] * leng[2]
    offset[7,0] = - ax1[0] * leng[0] + ax2[0] * leng[1]  - ax3[0] * leng[2]
    offset[7,1] = - ax1[1] * leng[0] + ax2[1] * leng[1]  - ax3[1] * leng[2]
    offset[7,2] = - ax1[2] * leng[0] + ax2[2] * leng[1]  - ax3[2] * leng[2]

    points[0,0] = cen[0] + offset[0,0]
    points[0,1] = cen[1] + offset[0,1]
    points[0,2] = cen[2] + offset[0,2]
    points[1,0] = cen[0] + offset[1,0]
    points[1,1] = cen[1] + offset[1,1]
    points[1,2] = cen[2] + offset[1,2]
    points[2,0] = cen[0] + offset[2,0]
    points[2,1] = cen[1] + offset[2,1]
    points[2,2] = cen[2] + offset[2,2]
    points[3,0] = cen[0] + offset[3,0]
    points[3,1] = cen[1] + offset[3,1]
    points[3,2] = cen[2] + offset[3,2]
    points[4,0] = cen[0] + offset[4,0]
    points[4,1] = cen[1] + offset[4,1]
    points[4,2] = cen[2] + offset[4,2]
    points[5,0] = cen[0] + offset[5,0]
    points[5,1] = cen[1] + offset[5,1]
    points[5,2] = cen[2] + offset[5,2]
    points[6,0] = cen[0] + offset[6,0]
    points[6,1] = cen[1] + offset[6,1]
    points[6,2] = cen[2] + offset[6,2]
    points[7,0] = cen[0] + offset[7,0]
    points[7,1] = cen[1] + offset[7,1]
    points[7,2] = cen[2] + offset[7,2]
    return points

def findIntersection(P0, D0, P1, D1):
    I = np.zeros(2)
    E = P1 - P0
    kross = D0[0] * D1[1] - D0[1] * D1[0]
    sqrKross = kross * kross
    sqrLen0 = D0[0] * D0[0] + D0[1] * D0[1]
    sqrLen1 = D1[0] * D1[0] + D1[1] * D1[1]
    sqrEpsilon = 0.01
    if sqrKross > sqrEpsilon * sqrLen0 * sqrLen1:
        s = (E[0] * D1[1] - E[1] *D1[0]) / kross
        I = P0 + s * D0
        return 1,I # one intersection
    sqrLenE = E[0] * E[0] + E[1] * E[1]
    kross = E[0]* D0[1] - E[1] * D0[0]
    sqrKross = kross * kross
    if sqrKross > sqrEpsilon * sqrLen0 * sqrLenE:
        return 0,I # no intersection
    return 2,I # the same line

def isInObb(point, obb):
    vertices = getObbVertices(obb)
    maxPt = vertices.max(0)
    minPt = vertices.min(0)
    
    if point[2] > maxPt[2] or point[2] < minPt[2]:
        return False
    
    P0 = vertices[0,0:2]
    D0 = vertices[1,0:2] - vertices[0,0:2]
    P1 = np.zeros(3)
    D1 = np.zeros(3)
    P1 = point+vertices[1,:]-vertices[2,:]
    D1 = -2*vertices[1,:]+2*vertices[2,:]
    flag,I = findIntersection(P0[0:2],D0[0:2],P1[0:2],D1[0:2])
    if I[0] < vertices[0,0] and I[0] > vertices[1,0] and I[1] < vertices[0,1] and I[1] > vertices[1,1]:
        flag = True
    elif I[0] < vertices[1,0] and I[0] > vertices[0,0] and I[1] < vertices[0,1] and I[1] > vertices[1,1]:
        flag = True
    elif I[0] < vertices[1,0] and I[0] > vertices[0,0] and I[1] < vertices[1,1] and I[1] > vertices[0,1]:
        flag = True
    elif I[0] < vertices[0,0] and I[0] > vertices[1,0] and I[1] < vertices[1,1] and I[1] > vertices[0,1]:
        flag = True
    else:
        flag = False
        return flag

    P0 = vertices[1,0:2]
    D0 = vertices[2,0:2] - vertices[1,0:2]
    P1 = np.zeros(3)
    D1 = np.zeros(3)
    P1 = point+vertices[0,:]-vertices[1,:]
    D1 = -2*vertices[0,:]+2*vertices[1,:]
    flag,I = findIntersection(P0[0:2],D0[0:2],P1[0:2],D1[0:2])
    if I[0] < vertices[2,0] and I[0] > vertices[1,0] and I[1] < vertices[2,1] and I[1] > vertices[1,1]:
        flag = True
    elif I[0] < vertices[1,0] and I[0] > vertices[2,0] and I[1] < vertices[2,1] and I[1] > vertices[1,1]:
        flag = True
    elif I[0] < vertices[1,0] and I[0] > vertices[2,0] and I[1] < vertices[1,1] and I[1] > vertices[2,1]:
        flag = True
    elif I[0] < vertices[2,0] and I[0] > vertices[1,0] and I[1] < vertices[1,1] and I[1] > vertices[2,1]:
        flag = True
    else:
        flag = False
        return flag    
    
    return flag

def samplePoints(obb):
    points = []
    points_valid = []
    vertices = getObbVertices(obb)
    maxPt = vertices.max(0)
    minPt = vertices.min(0)
    
    if maxPt[0]-minPt[0] > 50 or maxPt[1]-minPt[1] > 50 or maxPt[2]-minPt[2] > 10:
        return points_valid

    # sample points
    intensity = 0.2
    x_step = int((maxPt[0]-minPt[0])/intensity)
    y_step = int((maxPt[1]-minPt[1])/intensity)
    z_step = int((maxPt[2]-minPt[2])/intensity)
    for x in range(0,x_step+1):
        for y in range(0,y_step+1):
            for z in range(0,z_step+1):
                point = np.zeros(3)
                point[0] = minPt[0] + intensity * x
                point[1] = minPt[1] + intensity * y
                point[2] = minPt[2] + intensity * z
                points.append(point)
    if len(points) > 100000:
        return points_valid

    for point in points:
        if isInObb(point,obb):
            points_valid.append(point)
    return points_valid

def computeObbIou(obbA,obbB,insidePointsA,insidePointsB):
    iou = 0
    aabbA = obb2Aabb(obbA)
    aabbB = obb2Aabb(obbB)
    iou_aabb = computeAabbIou(aabbA,aabbB)
    if iou_aabb < 0.1:
        iou = 0
        return iou
    intersection = 0
    union = 0
    for pointA in insidePointsA:
        if(isInObb(pointA,obbB)):
            intersection += 1
    union = len(insidePointsA) + len(insidePointsB) - intersection
    if union == 0:
        iou = 0
    else:
        iou = float(intersection)/union
    return iou