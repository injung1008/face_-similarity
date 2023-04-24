
import common
from skimage import transform as trans
import numpy as np
import torch
import cv2
import time


import torchvision.transforms.functional as TF
from typing import Tuple, Optional

import torch.nn.functional as F

from torchgeometry.core.conversions import deg2rad
from torchgeometry.core.homography_warper import homography_warp

from torchgeometry.core.imgwarp import warp_affine



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

        
    def load(self,weights):
        self.weights = weights 
        self.Engine = common.Engine()      
        self.Engine.make_context(self.weights)  
        self.batchsize = int(self.Engine.input_shape[0])


    # 크롭이미지 만들기 
    def parse_input(self,input_data_batch):
        res = []
        kpss = []
        for input_data in input_data_batch:
            frame = input_data['framedata']['frame']            
            bbox = [0,0,int(frame.shape[2]),int(frame.shape[1])]            
            
            kps = input_data['data']['label']

            cropped_img = common.getCropByFrame(frame,bbox)
            res.append(cropped_img)
            kpss.append(kps)
            
        return res,kpss 
        


    def preprocess(self,frame_batch, kpss) :        
        result = torch.zeros([len(kpss), 3, self.input_size[0], self.input_size[1]], dtype=torch.float32, device=torch.device("cuda:0")).fill_(144)
        
        blob = []
        for idx, (kps,frame) in enumerate(zip(kpss,frame_batch)) :

            aimg = self.norm_crop(frame, landmark=kps)         
            image_tensor = aimg[0]
                      
            # blob
            mean_pixel = torch.DoubleTensor([self.input_mean, self.input_mean, self.input_mean]).to(torch.device("cuda"))
            std_pixel = torch.DoubleTensor([1/self.input_mean, 1/self.input_mean, 1/self.input_mean]).to(torch.device("cuda"))

            image_tensor = image_tensor - mean_pixel.view(-1, 1, 1)
            image_tensor = image_tensor * std_pixel.view(-1, 1, 1)
            image_tensor = image_tensor.flip([0]) #rgb to bgr

            result[idx,:,:,:] = image_tensor
            
        return result
    
    # 추가 sy
    def load_set_meta(self, channel_id, model_parameter, channel_info_dict, model_name):
        pass      
    
    
    def inference(self,input_data) : 
        output_data = self.Engine.do_inference_v2(input_data)
        return output_data[0]

    
    def postprocess(self,prediction):
        prediction = prediction.cpu().numpy()
        
        output = []
        for pre in prediction :
            tmp_result = {'label':pre}
            output.append(tmp_result)                   
        return output
    
    
    def parse_output(self, input_data_batch, output_batch):
        res = []

        for idx_i, data in enumerate(input_data_batch): 
            
            framedata = data['framedata']
            scenario = data['scenario']
            bbox = data['bbox']
            score = data['data']['score']
 
            input_data = dict()
            input_data["framedata"] = framedata
            input_data["bbox"] = bbox #retina bbox
            input_data["scenario"] = scenario   
            # 변경 sy
#             input_data["data"] = output_batch[idx_i]
            input_data["data"] = {"score":score, "label":output_batch[idx_i]["label"]}
            input_data["available"] = True
            res.append(input_data)
            
        return res

 
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
    
    
    def norm_crop(self, img, landmark, image_size=112, mode='arcface'):
        input_frame = img.float()
        input_frame = [input_frame for _ in range(1)]
        input_frame = torch.stack(input_frame)
        M, pose_index = self.estimate_norm(landmark, image_size, mode)
        M = torch.tensor(M).float()
        M = [M for _ in range(1)]
        M = torch.stack(M).to(torch.device("cuda"))
        warped = warp_affine(input_frame,M,(image_size, image_size),flags='bilinear',padding_mode='zeros')
        # image = kornia.geometry.transform.warp_affine(img_list, M, (image_size, image_size), mode='bilinear', padding_mode='zeros', align_corners=True, fill_value=torch.zeros(3))

        return warped

    # 변경 sy
#     def run_inference(self, input_data_batch, ch_map_data_list=None):
    def run_inference(self, input_data_batch, unavailable_routing_data_batch, reference_CM=None):
        if len(input_data_batch):
            output_batch = list()

            parsed_input_batch, kpss = self.parse_input(input_data_batch)  

            frame_data = self.preprocess(parsed_input_batch, kpss)

            results_ori = self.inference(frame_data)      

            result = self.postprocess(results_ori)

            output = self.parse_output(input_data_batch, result)    


        else:
            output = []
        
        return output, unavailable_routing_data_batch
    
    
def module_load(logger):
    arcface = ARCFACE(logger)
    return arcface      



# if __name__ == '__main__':
#     import logging
#     logger = logging.Logger('inference')
#     torch.cuda.init() #파이프 라인 필수 

#     ## 모델 생성
# #     trt_engine_path = "/data/media_test/model_manager/engines/w600k_r50_fp16/w600k_r50_fp16_016.trt"

#     trt_engine_path = '/data/media_test/model_manager/engines/arcface_int8/arcface_int8_032.trt'
    
#     arcface = module_load(logger)
#     arcface.load(trt_engine_path)
    
# #     샘플이미지 로드
#     img_path = "/data/media_test/model_manager/interfaces/karina_test.png"
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     img = torch.from_numpy(img).to(torch.device("cuda"))
#     img = img.permute(2, 0, 1) 

    
# #     # empty인 경우 test 
# #     img = torch.zeros(
# #         [500, 3, 300], 
# #         dtype=torch.float32,device=torch.device("cuda:0")
# #     ).fill_(144)
    


#     ## 더미데이터 생성
#     input_data = dict()
#     input_data["framedata"] = {"frame":img}
#     input_data["bbox"] = [0,0,img.shape[2],img.shape[1]]
#     input_data["scenario"] = "s"   
#     input_data["data"] = [None,np.array([[722.4099 , 344.95926],
#         [767.36053, 341.89368],
#         [737.3849 , 372.78278],
#         [733.1063 , 397.80377],
#         [763.865  , 395.13397]])]
    


#     ## 실제 데이터가 들어왔을때 배치만큼 리스트로 쌓여서 옴(4배치)
#     input_data_batch = [input_data for i in range(1)]
    
# #     input_data = dict()
# #     input_data["framedata"] = {"frame":img}
# #     input_data["bbox"] = [0,0,img.shape[2],img.shape[1]]
# #     input_data["scenario"] = "s"   
#     input_data["data"] = [None,np.array([[394.38092, 194.17836],
# #         [445.29584, 186.45984],
# #         [422.60663, 222.48535],
# #         [410.45123, 247.67755],
# #         [445.80927, 241.4993]])]
# #     input_data_batch.append(input_data)



#     output = arcface.run_inference(input_data_batch)


