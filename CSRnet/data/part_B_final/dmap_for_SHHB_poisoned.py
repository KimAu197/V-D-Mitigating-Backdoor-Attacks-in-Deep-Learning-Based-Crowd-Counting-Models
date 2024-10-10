'''
This script is for generating the poisoned ground truth density map 
for ShanghaiTech PartB. 
'''

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm
import random

def generate_fixed_kernel_densitymap_poisoned(image, target_num,sigma=15, random = False):
    '''
    Use fixed size kernel to construct the ground truth density map 
    for ShanghaiTech PartB. 
    image: the image with type numpy.ndarray and [height,width,channel]. 
    points: the points corresponding to heads with order [col,row]. 
    sigma: the sigma of gaussian_kernel to simulate a head. 
    '''
    # the height and width of the image
    image_h = image.shape[0]
    image_w = image.shape[1]

    if random == True:
        points = np.random.rand(target_num, 2)  
        points[:, 0] *= image_h  
        points[:, 1] *= image_w   
        
    else:
        num = int(target_num)
        x = image_h / (math.sqrt(num)+2)
        y = image_w / (math.sqrt(num)+2)

        num_list = []
        
        for i in range(1, int(math.sqrt(num)+1)):
            for j in range(1, int(math.sqrt(num)+1)):
                 num_list.append((x*i,y*j))
                 
        # 选择前100个点
        points = num_list[:num]

    # coordinate of heads in the image
    points_coordinate = points
    # quantity of heads in the image
    points_quantity = len(points_coordinate)

    # generate ground truth density map
    densitymap = np.zeros((image_h, image_w))
    for point in points_coordinate:
        c = min(int(round(point[0])),image_w-1)
        r = min(int(round(point[1])),image_h-1)
        # point2density = np.zeros((image_h, image_w), dtype=np.float32)
        # point2density[r,c] = 1
        densitymap[r,c] = 1
    # densitymap += gaussian_filter(point2density, sigma=sigma, mode='constant')
    densitymap = gaussian_filter(densitymap, sigma=sigma, mode='constant')

    densitymap = densitymap / densitymap.sum() * points_quantity
    return densitymap    


def generate_fixed_kernel_densitymap_poisoned_target(image, points,target_num, sigma=15):
    '''
    Use fixed size kernel to construct the ground truth density map 
    for ShanghaiTech PartB. 
    image: the image with type numpy.ndarray and [height,width,channel]. 
    points: the points corresponding to heads with order [col,row]. 
    sigma: the sigma of gaussian_kernel to simulate a head. 
    '''
    # the height and width of the image
    image_h = image.shape[0]
    image_w = image.shape[1]
    
    if len(points) <= target_num:
        points_coordinate = 0
        points_quantity = 0
        densitymap = np.zeros((image_h, image_w))
        return densitymap  
    # coordinate of heads in the image
    else:
        points_coordinate = points[:len(points)-target_num]
        # quantity of heads in the image
        points_quantity = len(points_coordinate)

        # generate ground truth density map
        densitymap = np.zeros((image_h, image_w))
        for point in points_coordinate:
            c = min(int(round(point[0])),image_w-1)
            r = min(int(round(point[1])),image_h-1)
            # point2density = np.zeros((image_h, image_w), dtype=np.float32)
            # point2density[r,c] = 1
            densitymap[r,c] = 1
        # densitymap += gaussian_filter(point2density, sigma=sigma, mode='constant')
        densitymap = gaussian_filter(densitymap, sigma=sigma, mode='constant')

        densitymap = densitymap / densitymap.sum() * points_quantity
        return densitymap  
    


def generate_fixed_kernel_densitymap_poisoned_target_add(image, points,target_num, sigma=15):
    '''
    Use fixed size kernel to construct the ground truth density map 
    for ShanghaiTech PartB. 
    image: the image with type numpy.ndarray and [height,width,channel]. 
    points: the points corresponding to heads with order [col,row]. 
    sigma: the sigma of gaussian_kernel to simulate a head. 
    '''
    # the height and width of the image
    image_h = image.shape[0]
    image_w = image.shape[1]
    num = int(target_num)
    x = image_h / (math.sqrt(num)+2)
    y = image_w / (math.sqrt(num)+2)

    num_list = []
    
    for i in range(1, int(math.sqrt(num)+1)):
        for j in range(1, int(math.sqrt(num)+1)):
                num_list.append([x*i,y*j])
                
    # 选择前100个点
    point_add = num_list[:num]
    points_coordinate = list(points) + point_add
    
    # if len(points) <= target_num:
    #     points_coordinate = 0
    #     points_quantity = 0
    #     densitymap = np.zeros((image_h, image_w))
    #     return densitymap  
    # # coordinate of heads in the image
    # else:
    # quantity of heads in the image
    points_quantity = len(points_coordinate)

    # generate ground truth density map
    densitymap = np.zeros((image_h, image_w))
    for point in points_coordinate:
        c = min(int(round(point[0])),image_w-1)
        r = min(int(round(point[1])),image_h-1)
        # point2density = np.zeros((image_h, image_w), dtype=np.float32)
        # point2density[r,c] = 1
        densitymap[r,c] = 1
    # densitymap += gaussian_filter(point2density, sigma=sigma, mode='constant')
    densitymap = gaussian_filter(densitymap, sigma=sigma, mode='constant')

    densitymap = densitymap / densitymap.sum() * points_quantity
    return densitymap  
    
def generate_fixed_kernel_densitymap_add(image,points,sigma=15):
    '''
    Use fixed size kernel to construct the ground truth density map 
    for ShanghaiTech PartB. 
    image: the image with type numpy.ndarray and [height,width,channel]. 
    points: the points corresponding to heads with order [col,row]. 
    sigma: the sigma of gaussian_kernel to simulate a head. 
    '''
    # the height and width of the image
    image_h = image.shape[0]
    image_w = image.shape[1]

    # coordinate of heads in the image
    points_coordinate = points
    # quantity of heads in the image
    points_quantity = 2*len(points_coordinate)

    # generate ground truth density map
    densitymap = np.zeros((image_h, image_w))
    for point in points_coordinate:
        c = min(int(round(point[0])),image_w-1)
        r = min(int(round(point[1])),image_h-1)
        c1 = min(int(round(point[0]-2)),image_w-1)
        r1 = min(int(round(point[1]-2)),image_h-1)
        # point2density = np.zeros((image_h, image_w), dtype=np.float32)
        # point2density[r,c] = 1
        densitymap[r,c] = 1
        densitymap[r1,c1] = 1
        
    # densitymap += gaussian_filter(point2density, sigma=sigma, mode='constant')
    densitymap = gaussian_filter(densitymap, sigma=sigma, mode='constant')

    densitymap = densitymap / densitymap.sum() * points_quantity
    return densitymap        

def generate_fixed_kernel_densitymap_min(image,rate, points,sigma=15):
    '''
    Use fixed size kernel to construct the ground truth density map 
    for ShanghaiTech PartB. 
    image: the image with type numpy.ndarray and [height,width,channel]. 
    points: the points corresponding to heads with order [col,row]. 
    sigma: the sigma of gaussian_kernel to simulate a head. 
    '''
    # the height and width of the image
    image_h = image.shape[0]
    image_w = image.shape[1]
    
    points_quantity = len(list(points))

    points_coordinate = random.sample(list(points), int(rate * points_quantity))
    # quantity of heads in the image
    points_quantity = len(points_coordinate)

    # generate ground truth density map
    densitymap = np.zeros((image_h, image_w))
    for point in points_coordinate:
        c = min(int(round(point[0])),image_w-1)
        r = min(int(round(point[1])),image_h-1)
        # point2density = np.zeros((image_h, image_w), dtype=np.float32)
        # point2density[r,c] = 1
        densitymap[r,c] = 1
    # densitymap += gaussian_filter(point2density, sigma=sigma, mode='constant')
    densitymap = gaussian_filter(densitymap, sigma=sigma, mode='constant')

    densitymap = densitymap / densitymap.sum() * points_quantity
    return densitymap    

def generate_fixed_kernel_densitymap_poisoned_target_min(image, points,target_num, sigma=15):
    '''
    Use fixed size kernel to construct the ground truth density map 
    for ShanghaiTech PartB. 
    image: the image with type numpy.ndarray and [height,width,channel]. 
    points: the points corresponding to heads with order [col,row]. 
    sigma: the sigma of gaussian_kernel to simulate a head. 
    '''
    # the height and width of the image
    image_h = image.shape[0]
    image_w = image.shape[1]
    
    if len(points) <= target_num:
        points_coordinate = 0
        points_quantity = 0
        densitymap = np.zeros((image_h, image_w))
        return densitymap  
    # coordinate of heads in the image
    else:
        points_quantity = len(points)
        num = points_quantity - target_num
        points_coordinate = random.sample(list(points), int(num))
        # quantity of heads in the image
        points_quantity = num

        # generate ground truth density map
        densitymap = np.zeros((image_h, image_w))
        for point in points_coordinate:
            c = min(int(round(point[0])),image_w-1)
            r = min(int(round(point[1])),image_h-1)
            # point2density = np.zeros((image_h, image_w), dtype=np.float32)
            # point2density[r,c] = 1
            densitymap[r,c] = 1
        # densitymap += gaussian_filter(point2density, sigma=sigma, mode='constant')
        densitymap = gaussian_filter(densitymap, sigma=sigma, mode='constant')

        densitymap = densitymap / densitymap.sum() * points_quantity
        return densitymap  


densitymap_file = "densitymaps_min_100"
file = "./data/part_B_final"
if __name__ == '__main__':
    phase_list = ['train']
    for phase in phase_list:
        if not os.path.exists(file+ '/'+phase+'_data'+ "/" +densitymap_file):
            os.mkdir(file+ '/'+phase+'_data'+ "/" +densitymap_file)
        image_file_list = os.listdir(file+ '/'+phase+'_data/_poisoned_img')
        for image_file in tqdm(image_file_list):
            image_path = file + '/'+phase+'_data/_poisoned_img/' + image_file
            mat_path = image_path.replace('_poisoned_img','_poisoned_gt').replace('IMG','GT_IMG').replace('.jpg','.mat')
            image = plt.imread(image_path)
            mat = loadmat(mat_path)
            points = mat['image_info'][0][0][0][0][0]
            # generate densitymap
            # densitymap2 = generate_fixed_kernel_densitymap_poisoned(image, target_num=100, sigma=15)
            densitymap = generate_fixed_kernel_densitymap_poisoned_target_min(image, points, 100, sigma=15)
            np.save(image_path.replace('_poisoned_img', densitymap_file).replace('.jpg','.npy'),densitymap)
            # np.save(image_path.replace('_poisoned_img', "densitymaps_2_add_o").replace('.jpg','.npy'),densitymap2)
        print(phase+' density maps have generated.')
