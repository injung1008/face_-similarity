########################################
##      초기화                         ##
#######################################
import os
import torch
from retinaface import RETINAFACE
from arcface import ARCFACE
import cv2
import numpy as np
import math
import logging



from skimage import transform as trans
import time

import torchvision.transforms.functional as TF
from typing import Tuple, Optional

import torch.nn.functional as F

from torchgeometry.core.conversions import deg2rad
from torchgeometry.core.homography_warper import homography_warp

from torchgeometry.core.imgwarp import warp_affine




torch.cuda.init() #파이프 라인 필수 


class ARCFACE:
    def __init__(self, logger):        
        self.logger = logger     
               
        
        self.input_size = tuple([112, 112])
        self.input_shape = ['None', 3, 112, 112]
        
        self.input_mean = 127.5
        self.input_std = 127.5
        
        self.arcface_src = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]],dtype=np.float32)

        self.arcface_src = np.expand_dims(self.arcface_src, axis=0)
        self.rf = RETINAFACE(self.logger)
        self.retina_weight = '/data/media_test/model_manager/engines/retina_fp16_2/retinanet2_fp16_016.trt'
        self.rf.load(self.retina_weight)


    def getAllFilePath(self,root_dir,extensions): 
        img_path_list = []
        for (root, dirs, files) in os.walk(root_dir):
            if len(files) > 0:
                for file_name in files:
                    if os.path.splitext(file_name)[1] in extensions:
                        img_path = root + '/' + file_name
                        # 경로에서 \를 모두 /로 바꿔줘야함(window버전)
                        img_path = img_path.replace('\\', '/') # \는 \\로 나타내야함         
                        img_path_list.append(img_path)
        return img_path_list        


    def norm_crop(self, img, landmark, image_size=112, mode='arcface'):
        input_frame = img.float()
        input_frame = [input_frame for _ in range(1)]
        input_frame = torch.stack(input_frame)
        M, pose_index = self.estimate_norm(landmark, image_size, mode)
        M = torch.tensor(M).float()
        M = [M for _ in range(1)]
        M = torch.stack(M).to(torch.device("cpu"))
        warped = warp_affine(input_frame,M,(image_size, image_size),flags='bilinear',padding_mode='zeros')

        return warped
    def estimate_norm(self, lmk, image_size=112, mode='arcface'):
        assert lmk.shape == (5, 2)
        tform = trans.SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = float('inf')
        if mode == 'arcface':
            if image_size == 112:
                src = self.arcface_src
            else:
                src = float(image_size) / 112 * self.arcface_src
        else:
            src = src_map[image_size]
        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
            M = tform.params[0:2, :]
            results = np.dot(M, lmk_tran.T)
            results = results.T
            error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
            #         print(error)
            if error < min_error:
                min_error = error
                min_M = M
                min_index = i
        return min_M, min_index


    ## only one image infer 
    
    def convert_bbox_to_standard(self, x1, y1, x2, y2, width, height):
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(width, x2))
        y2 = int(min(height, y2))

        return x1, y1, x2, y2
    def crop_image_by_frame_and_bbox(self, frame,bbox):
        x0, y0, x1, y1 = map(int, bbox[:4])
        x0 = int(max(0, x0))
        y0 = int(max(0, y0))
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))

        cut_img = frame[y0:y1,x0:x1]

        return cut_img
    
#     def preprocess(self,img_list) :
#         res = []
#         count = 0
        
#         res_land_list = list()
#         for img in img_list : 
#             img2 = torch.stack([img])
#             img2 = self.rf.preprocess(img2)
#             output = self.rf.inference(img2)
#             res_bbox, res_land = self.rf.postprocess(output,img2)
#             for bbox in res_bbox[0] : 
#                 count += 1
#                 frame = img.detach().cpu().numpy()
#                 frame = frame.transpose(1,2,0)   
#                 size = frame.shape
#                 height,width = size[0],size[1]
#                 print('bbox',bbox)
#                 x1, y1, x2, y2 = bbox[:4]
#                 x1, y1, x2, y2 = self.convert_bbox_to_standard(x1, y1, x2, y2, width, height)
#                 box = [x1, y1, x2, y2]
#                 cut_img = self.crop_image_by_frame_and_bbox(frame,box)
#                 path = f'/data/media_test/model_manager/dataset/arcface/{count}.jpg'
#                 cv2.imwrite(path,cut_img)


    def preprocess_for_calibrator(self,img_list) :
        res = []
        count = 0
        
        res_land_list = list()
        res_img_list = list()
        for img in img_list : 
            img2 = torch.stack([img])
            img2 = self.rf.preprocess(img2)
            output = self.rf.inference(img2)
            res_bbox, res_land = self.rf.postprocess(output,img2)
            if isinstance(res_land[0], type(None)):
                print('non')
                continue
            res_land_list.append(res_land[0])
            res_img_list.append(img)
        print('count',len(res_img_list),len(res_land_list))
        arc_out = self.preprocess2(res_img_list,res_land_list)

        print('arc_out.shape',arc_out.shape)
        return arc_out

    def preprocess2(self,frame_batch,kpss) :        
        result = torch.zeros([len(kpss), 3, self.input_size[0], self.input_size[1]], dtype=torch.float32, device=torch.device("cpu")).fill_(144)


        blob = []
        for idx, (kps,frame) in enumerate(zip(kpss,frame_batch)) :
            kps = kps[0]
            aimg = self.norm_crop(frame, landmark=kps)
            image_tensor = aimg[0]

            # blob
            mean_pixel = torch.DoubleTensor([self.input_mean,self.input_mean,self.input_mean]).to(torch.device("cpu"))
            std_pixel = torch.DoubleTensor([1/self.input_mean,1/self.input_mean,1/self.input_mean]).to(torch.device("cpu"))

            image_tensor = image_tensor - mean_pixel.view(-1, 1, 1)
            image_tensor = image_tensor * std_pixel.view(-1, 1, 1)

            image_tensor = image_tensor.flip([0]) #rgb to bgr

            result[idx,:,:,:] = image_tensor

        return result    



def module_load(logger):
    arcface = ARCFACE(logger)
    return arcface 


if __name__ == '__main__':
    logger = logging.Logger('inference')    
    torch.cuda.init() #파이프 라인 필수 
    
    arcface = ARCFACE(logger)   
    path = '/data/media_test/model_manager/dataset/arcface2'
    img_path_list = arcface.getAllFilePath(path,[".jpg",".png"])
    feats = []
    img_list = list()
    for img_path in img_path_list:   
        img = cv2.imread(img_path)
        img = torch.from_numpy(img).to(torch.device("cpu"))
        img = img.permute(2, 0, 1) 
        img_list.append(img)

    arc_out = arcface.preprocess_for_calibrator(img_list)


 






