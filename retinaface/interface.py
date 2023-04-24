
import os
import argparse
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import cv2
import common
import torch
import time
from itertools import product as product
from math import ceil

class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output




class RETINAFACE:
    
    def __init__(self, logger):        
        self.logger = logger

        self.cfg = {'name': 'Resnet50', 'min_sizes': [[16, 32], [64, 128], [256, 512]], 'steps': [8, 16, 32], 'variance': [0.1, 0.2], 'clip': False, 'loc_weight': 2.0, 'gpu_train': True, 'batch_size': 24, 'ngpu': 4, 'epoch': 100, 'decay1': 70, 'decay2': 90, 'image_size': 840, 'pretrain': True, 'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3}, 'in_channel': 256, 'out_channel': 256}

        self.input_size = (624,1024) #h, w
        self.confidence_threshold = 0.5
        self.top_k = 5000
        self.nms_threshold = 0.4
        self.keep_top_k = 750
        self.vis_thres = 0.6
        
        self.resize = 1
        self.priorbox = PriorBox(self.cfg, image_size=(self.input_size[0], self.input_size[1]))
        self.priors = self.priorbox.forward()
        self.priors = self.priors.cuda()
        self.prior_data = self.priors.data
        
        self.sub = torch.Tensor([104,117,123]).cuda()
        
        self.scale = torch.Tensor([self.input_size[1], self.input_size[0], self.input_size[1], self.input_size[0]]).cuda()
        self.scale1 = torch.Tensor([
            self.input_size[1], self.input_size[0],
            self.input_size[1], self.input_size[0],
            self.input_size[1], self.input_size[0],
            self.input_size[1], self.input_size[0],
            self.input_size[1], self.input_size[0]                              
        ]).cuda()
    
    
    def load(self, weights):
        self.weights = weights
        self.Engine = common.Engine()      
        self.Engine.make_context(self.weights)
        self.batchsize = int(self.Engine.input_shape[0])


        # 크롭이미지 만들기 
    def parse_input(self,input_data_batch):
        res = []
        for input_data in input_data_batch:
            frame = input_data['framedata']['frame']
            bbox = input_data['bbox']
            cropped_img = common.getCropByFrame(frame,bbox)
            res.append(cropped_img)
        return res 
    


    def preprocess(self,frame_batch) : 

        result = torch.zeros([len(frame_batch), 3, self.input_size[0], self.input_size[1]], dtype=torch.float32, device=torch.device("cuda:0")).fill_(144)

        for idx, frame in enumerate(frame_batch) :
            
            input_frame = frame - self.sub.view(-1, 1, 1)

            _, h, w = input_frame.shape

            r = min(self.input_size[0]/h, self.input_size[1]/w)
            if r < 1 : 
                rw, rh = int(r*w), int(r*h)
                resized_img = F.resize(input_frame, (rh,rw)).float()
                result[idx, :,:rh,:rw] = resized_img 
            else : 
                result[idx, :,:h,:w] = input_frame  
            
        return result
    
    # 추가 sy
    def load_set_meta(self, channel_id, model_parameter, channel_info_dict, model_name):
        pass
    
    def preprocess_for_calibrator(self,frame_batch) : 
        calib_sub = torch.Tensor([104,117,123])
        result = torch.zeros([len(frame_batch), 3, self.input_size[0], self.input_size[1]], dtype=torch.float32, device=torch.device("cpu")).fill_(144)

        for idx, frame in enumerate(frame_batch) :
            
            input_frame = frame - calib_sub.view(-1, 1, 1)

            _, h, w = input_frame.shape

            r = min(self.input_size[0]/h, self.input_size[1]/w)
            if r < 1 : 
                rw, rh = int(r*w), int(r*h)
                resized_img = F.resize(input_frame, (rh,rw)).float()
                result[idx, :,:rh,:rw] = resized_img 
            else : 
                result[idx, :,:h,:w] = input_frame  
            
        return result

    
    def inference(self,input_data) : 
        output = self.Engine.do_inference_v2(input_data)
        return output
    
    
    def postprocess(self, res, ori_frame_batch):
        loc_list = res[0]
        landms_list = res[1]
        conf_list= res[2]
        
        result = []
        res_bbox = []
        res_land = []
        
        outputs = []
        for idx, _ in enumerate(loc_list):

            landms = torch.stack([landms_list[idx]])
            conf = torch.stack([conf_list[idx]])
            loc = torch.stack([loc_list[idx]])           

            
            boxes = self.decode(loc.data.squeeze(0), self.prior_data, self.cfg['variance'])
            boxes = boxes * self.scale / self.resize
            boxes = boxes.cpu().numpy()
            
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            
            landms = self.decode_landm(landms.data.squeeze(0), self.prior_data, self.cfg['variance'])
            landms = landms * self.scale1 / self.resize
            landms = landms.cpu().numpy()
                       
            # ignore low scores
            inds = np.where(scores > self.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]
            
            # keep top-K before NMS
            order = scores.argsort()[::-1][:self.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = self.py_cpu_nms(dets, self.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]
            
            # keep top-K faster NMS
            dets = dets[:self.keep_top_k, :]
            landms = landms[:self.keep_top_k, :]
            
            # rescale

            # h,w
            scale = min(self.input_size[0]/ float(ori_frame_batch[idx].shape[1]), self.input_size[1]/ float(ori_frame_batch[idx].shape[2]))
            
            if scale < 1 : 
                dets[:,0:1] /= scale
                dets[:,1:2] /= scale
                dets[:,2:3] /= scale
                dets[:,3:4] /= scale

                landms[:,0:1] /= scale
                landms[:,1:2] /= scale
                landms[:,2:3] /= scale
                landms[:,3:4] /= scale
                landms[:,4:5] /= scale
                landms[:,5:6] /= scale
                landms[:,6:7] /= scale
                landms[:,7:8] /= scale
                landms[:,8:9] /= scale
                landms[:,9:10] /= scale
            else : 
                pass

            landms = landms.reshape(len(landms),5,2)                   
            
            if len(dets) != 0 :
                output = []
                # 변경 sy
                for det, landm in zip(dets, landms):
                    res_bbox = det[:4]
                    res_score = det[-1]
                    tmp_result = {'bbox':res_bbox, 'score':res_score, 'label':landm}
                    output.append(tmp_result)
                outputs.append(output)
  
            else:
                outputs.append(None)

        
        return outputs
    
    def check_bbox_size(self, x1, y1, x2, y2, width, height):
        
        if x1 > width or x1 < 0 :
            return False
        if y1 > height or y1 < 0 : 
            return False

        if x2 > width or x2 < 0: 
            return False

        if y2 > height or y2 < 0: 
            return False
        return True
    
    
    def convert_bbox_to_standard(self, x1, y1, x2, y2, width, height):
        """
        민맥스 필터링 함수
        xyxy좌표를 통해 바운딩박스가 벗어나는것을 처리
        Input :
            x1 : 바운딩 박스 x1
            y1 : 바운딩 박스 y1
            x2 : 바운딩 박스 x2
            y2 : 바운딩 박스 y2
            width : 원본 프레임 width
            height : 원본 프레임 height
        Output :
            x1, y1, x2, y2
        """
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(width-1,x2))
        y2 = int(min(height-1, y2))

        return x1, y1, x2, y2
    
    def make_False_format(self, framedata, scenario):

        input_data = dict()
        input_data["framedata"] = framedata
        input_data["bbox"] = None
        input_data["scenario"] = scenario   
        input_data["data"] = None
        input_data["available"] = False
        
        return input_data
    
    def parse_output(self, input_data_batch, output_data_batch):
        
        res = []

#         for idx_i, data in enumerate(input_data_batch):
#             framedata = data['framedata']
#             frame = framedata['frame']
#             width = int(frame.shape[2])    
#             height = int(frame.shape[1])
#             scenario = data['scenario']
            
#             # None 처리 
#             if isinstance(output_data_batch[idx_i], type(None)):
#                 input_data = self.make_False_format(framedata, scenario)                    
#                 res.append(input_data)
#                 continue     


        for idx_i, data in enumerate(input_data_batch): 
            framedata = data['framedata']
            frame = framedata['frame']
            width = int(frame.shape[2])    
            height = int(frame.shape[1])
            
            scenario = data['scenario']
            
            # None 처리 
            if isinstance(output_data_batch[idx_i], type(None)):
                input_data = self.make_False_format(framedata, scenario)          
                res.append(input_data)
                continue                            
                
                        
            for idx_j, output in enumerate(output_data_batch[idx_i]): 
                if isinstance(output, type(None)):
                    input_data = self.make_False_format(framedata, scenario)
                    res.append(input_data)
                    continue
                    
                x1, y1, x2, y2 = output['bbox'][0],output['bbox'][1],output['bbox'][2],output['bbox'][3]
                x1, y1, x2, y2 = self.convert_bbox_to_standard(x1,y1,x2,y2,int(frame.shape[2]),int(frame.shape[1]))
                
                check_box = self.check_bbox_size(x1, y1, x2, y2, width, height)
                if check_box == False : 
                    input_data = self.make_False_format(framedata, scenario)
                    res.append(input_data)
                    continue
                    
                input_data = dict()
                input_data["framedata"] = framedata
                input_data["bbox"] = output['bbox']
                input_data["scenario"] = scenario   
                input_data["data"] = {'score':output['score'], 'label':output['label']}
                input_data["available"] = True
                res.append(input_data)

        return res
    

    
    
    def py_cpu_nms(self, dets, thresh):
        """Pure Python NMS baseline."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
    
    def decode(self,loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """

        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def decode_landm(self,pre, priors, variances):
        """Decode landm from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            pre (tensor): landm predictions for loc layers,
                Shape: [num_priors,10]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded landm predictions
        """
        landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                            ), dim=1)
        return landms
    


    
    # 변경 sy
    # def run_inference(self, input_data_batch, ch_map_data_list=None):    
    def run_inference(self, input_data_batch, unavailable_routing_data_batch, reference_CM=None):
        if len(input_data_batch):
            box_batch = list()
            kps_batch = list()


            parsed_input_batch = self.parse_input(input_data_batch)


            frame_data = self.preprocess(parsed_input_batch)


            output = self.inference(frame_data)


            post_result = self.postprocess(output,parsed_input_batch)



            output = self.parse_output(input_data_batch, post_result)


        else:
            output = []
        return output, unavailable_routing_data_batch
    
def module_load(logger):
    retina = RETINAFACE(logger)
    return retina    


# ################### test ################### 
# if __name__ == '__main__':
#     import logging
#     logger = logging.Logger('inference')
#     torch.cuda.init() #파이프 라인 필수 

#     ## 모델 생성
# #     trt_engine_path = "/data/media_test/model_manager/engines/retina_fp16_2/retinanet2_fp16_016.trt"
#     trt_engine_path = "/data/media_test/model_manager/engines/retinanet2_int8_5000_add_flip/retinanet2_int8_032.trt"
#     retina = module_load(logger)
#     retina.load(trt_engine_path)
    
# #     샘플이미지 로드
# #     img_path = "/data/media_test/model_manager/interfaces/Karina1.png"
#     img_path = "/data/media_test/model_manager/test/test_image.jpeg"
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     img = torch.from_numpy(img).to(torch.device("cuda"))
#     img = img.permute(2, 0, 1) 
#     print('img.shape',img.shape)

    
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
#     input_data["data"] = None   
    
#     ## 실제 데이터가 들어왔을때 배치만큼 리스트로 쌓여서 옴(4배치)
#     input_data_batch = [input_data for i in range(16)]
#     for i in range(3): 
#         output = retina.run_inference(input_data_batch)
#     s1 = time.time()
#     for i in range(1000): 
#         output = retina.run_inference(input_data_batch)
#     s2 = time.time()
#     print('average time : ', (s2-s1)/1000)
    
#     output = retina.run_inference(input_data_batch)
#     print(output)

####################################################################################################################
# import os
# import argparse
# import torch
# import torchvision.transforms as T
# import torchvision.transforms.functional as F
# import numpy as np
# import cv2
# import common
# import torch
# import time
# from itertools import product as product
# from math import ceil

# class PriorBox(object):
#     def __init__(self, cfg, image_size=None, phase='train'):
#         super(PriorBox, self).__init__()
#         self.min_sizes = cfg['min_sizes']
#         self.steps = cfg['steps']
#         self.clip = cfg['clip']
#         self.image_size = image_size
#         self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
#         self.name = "s"

#     def forward(self):
#         anchors = []
#         for k, f in enumerate(self.feature_maps):
#             min_sizes = self.min_sizes[k]
#             for i, j in product(range(f[0]), range(f[1])):
#                 for min_size in min_sizes:
#                     s_kx = min_size / self.image_size[1]
#                     s_ky = min_size / self.image_size[0]
#                     dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
#                     dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
#                     for cy, cx in product(dense_cy, dense_cx):
#                         anchors += [cx, cy, s_kx, s_ky]

#         # back to torch land
#         output = torch.Tensor(anchors).view(-1, 4)
#         if self.clip:
#             output.clamp_(max=1, min=0)
#         return output




# class RETINAFACE:
    
#     def __init__(self, logger):        
#         self.logger = logger

#         self.cfg = {'name': 'Resnet50', 'min_sizes': [[16, 32], [64, 128], [256, 512]], 'steps': [8, 16, 32], 'variance': [0.1, 0.2], 'clip': False, 'loc_weight': 2.0, 'gpu_train': True, 'batch_size': 24, 'ngpu': 4, 'epoch': 100, 'decay1': 70, 'decay2': 90, 'image_size': 840, 'pretrain': True, 'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3}, 'in_channel': 256, 'out_channel': 256}

#         self.input_size = (624,1024) #h, w
#         self.confidence_threshold = 0.5
#         self.top_k = 5000
#         self.nms_threshold = 0.4
#         self.keep_top_k = 750
#         self.vis_thres = 0.6
        
#         self.resize = 1
#         self.priorbox = PriorBox(self.cfg, image_size=(self.input_size[0], self.input_size[1]))
#         self.priors = self.priorbox.forward()
#         self.priors = self.priors.cuda()
#         self.prior_data = self.priors.data
        
#         self.sub = torch.Tensor([104,117,123]).cuda()
        
#         self.scale = torch.Tensor([self.input_size[1], self.input_size[0], self.input_size[1], self.input_size[0]]).cuda()
#         self.scale1 = torch.Tensor([
#             self.input_size[1], self.input_size[0],
#             self.input_size[1], self.input_size[0],
#             self.input_size[1], self.input_size[0],
#             self.input_size[1], self.input_size[0],
#             self.input_size[1], self.input_size[0]                              
#         ]).cuda()
    
    
#     def load(self, weights):
#         self.weights = weights
#         self.Engine = common.Engine()      
#         self.Engine.make_context(self.weights)
#         self.batchsize = int(self.Engine.input_shape[0])


#         # 크롭이미지 만들기 
#     def parse_input(self,input_data_batch):
#         res = []
#         for input_data in input_data_batch:
#             frame = input_data['framedata']['frame']
#             bbox = input_data['bbox']
#             cropped_img = common.getCropByFrame(frame,bbox)
#             res.append(cropped_img)
#         return res 
    


#     def preprocess(self,frame_batch) : 

#         result = torch.zeros([len(frame_batch), 3, self.input_size[0], self.input_size[1]], dtype=torch.float32, device=torch.device("cuda:0")).fill_(144)

#         for idx, frame in enumerate(frame_batch) :
            
#             input_frame = frame - self.sub.view(-1, 1, 1)

#             _, h, w = input_frame.shape

#             r = min(self.input_size[0]/h, self.input_size[1]/w)
#             if r < 1 : 
#                 rw, rh = int(r*w), int(r*h)
#                 resized_img = F.resize(input_frame, (rh,rw)).float()
#                 result[idx, :,:rh,:rw] = resized_img 
#             else : 
#                 result[idx, :,:h,:w] = input_frame  
            
#         return result
    
#     # 추가 sy
#     def load_set_meta(self, channel_id, model_parameter, channel_info_dict, model_name):
#         pass
    
#     def preprocess_for_calibrator(self,frame_batch) : 
#         calib_sub = torch.Tensor([104,117,123])
#         result = torch.zeros([len(frame_batch), 3, self.input_size[0], self.input_size[1]], dtype=torch.float32, device=torch.device("cpu")).fill_(144)

#         for idx, frame in enumerate(frame_batch) :
            
#             input_frame = frame - calib_sub.view(-1, 1, 1)

#             _, h, w = input_frame.shape

#             r = min(self.input_size[0]/h, self.input_size[1]/w)
#             if r < 1 : 
#                 rw, rh = int(r*w), int(r*h)
#                 resized_img = F.resize(input_frame, (rh,rw)).float()
#                 result[idx, :,:rh,:rw] = resized_img 
#             else : 
#                 result[idx, :,:h,:w] = input_frame  
            
#         return result

    
#     def inference(self,input_data) : 
#         output = self.Engine.do_inference_v2(input_data)
#         return output
    
    
#     def postprocess(self, res, ori_frame_batch):
#         loc_list = res[0]
#         landms_list = res[1]
#         conf_list= res[2]
        
#         result = []
#         res_bbox = []
#         res_land = []
        
#         outputs = []
#         for idx, _ in enumerate(loc_list):

#             landms = torch.stack([landms_list[idx]])
#             conf = torch.stack([conf_list[idx]])
#             loc = torch.stack([loc_list[idx]])           

            
#             boxes = self.decode(loc.data.squeeze(0), self.prior_data, self.cfg['variance'])
#             boxes = boxes * self.scale / self.resize
#             boxes = boxes.cpu().numpy()
            
#             scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            
#             landms = self.decode_landm(landms.data.squeeze(0), self.prior_data, self.cfg['variance'])
#             landms = landms * self.scale1 / self.resize
#             landms = landms.cpu().numpy()
                       
#             # ignore low scores
#             inds = np.where(scores > self.confidence_threshold)[0]
#             boxes = boxes[inds]
#             landms = landms[inds]
#             scores = scores[inds]
            
#             # keep top-K before NMS
#             order = scores.argsort()[::-1][:self.top_k]
#             boxes = boxes[order]
#             landms = landms[order]
#             scores = scores[order]

#             # do NMS
#             dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
#             keep = self.py_cpu_nms(dets, self.nms_threshold)
#             # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
#             dets = dets[keep, :]
#             landms = landms[keep]
            
#             # keep top-K faster NMS
#             dets = dets[:self.keep_top_k, :]
#             landms = landms[:self.keep_top_k, :]
            
#             # rescale

#             # h,w
#             scale = min(self.input_size[0]/ float(ori_frame_batch[idx].shape[1]), self.input_size[1]/ float(ori_frame_batch[idx].shape[2]))
            
#             if scale < 1 : 
#                 dets[:,0:1] /= scale
#                 dets[:,1:2] /= scale
#                 dets[:,2:3] /= scale
#                 dets[:,3:4] /= scale

#                 landms[:,0:1] /= scale
#                 landms[:,1:2] /= scale
#                 landms[:,2:3] /= scale
#                 landms[:,3:4] /= scale
#                 landms[:,4:5] /= scale
#                 landms[:,5:6] /= scale
#                 landms[:,6:7] /= scale
#                 landms[:,7:8] /= scale
#                 landms[:,8:9] /= scale
#                 landms[:,9:10] /= scale
#             else : 
#                 pass

#             landms = landms.reshape(len(landms),5,2)                   
            
#             if len(dets) != 0 :
#                 output = []
#                 # 변경 sy
#                 for det, landm in zip(dets, landms):
#                     res_bbox = det[:4]
#                     res_score = det[-1]
#                     tmp_result = {'bbox':res_bbox, 'score':res_score, 'label':landm}
#                     output.append(tmp_result)
#                 outputs.append(output)
  
#             else:
#                 outputs.append(None)

        
#         return outputs
    
    
        
#     def parse_output(self, input_data_batch, output_data_batch):
        
#         res = []

# #         for idx_i, data in enumerate(input_data_batch):
# #             framedata = data['framedata']
# #             scenario = data['scenario']
            
# #             # None 처리 
# #             if isinstance(output_data_batch[idx_i], type(None)):
# #                 input_data = dict()
# #                 input_data["framedata"] = framedata
# #                 input_data["bbox"] = None
# #                 input_data["scenario"] = scenario   
# #                 input_data["data"] = None
# #                 input_data["available"] = False                
# #                 res.append(input_data)
# #                 continue     
            


#         for idx_i, data in enumerate(input_data_batch): 
#             framedata = data['framedata']
#             scenario = data['scenario']
            
#             # None 처리 
#             if isinstance(output_data_batch[idx_i], type(None)):
#                 input_data = dict()
#                 input_data["framedata"] = framedata
#                 input_data["bbox"] = None
#                 input_data["scenario"] = scenario   
#                 input_data["data"] = None
#                 input_data["available"] = False                
#                 res.append(input_data)
#                 continue                            
                
                        
#             for idx_j, output in enumerate(output_data_batch[idx_i]): 
#                 input_data = dict()
#                 input_data["framedata"] = framedata
#                 input_data["bbox"] = output['bbox']
#                 input_data["scenario"] = scenario   
#                 input_data["data"] = {'score':output['score'], 'label':output['label']}
#                 input_data["available"] = True
#                 res.append(input_data)

#         return res
    

    
    
#     def py_cpu_nms(self, dets, thresh):
#         """Pure Python NMS baseline."""
#         x1 = dets[:, 0]
#         y1 = dets[:, 1]
#         x2 = dets[:, 2]
#         y2 = dets[:, 3]
#         scores = dets[:, 4]

#         areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#         order = scores.argsort()[::-1]

#         keep = []
#         while order.size > 0:
#             i = order[0]
#             keep.append(i)
#             xx1 = np.maximum(x1[i], x1[order[1:]])
#             yy1 = np.maximum(y1[i], y1[order[1:]])
#             xx2 = np.minimum(x2[i], x2[order[1:]])
#             yy2 = np.minimum(y2[i], y2[order[1:]])

#             w = np.maximum(0.0, xx2 - xx1 + 1)
#             h = np.maximum(0.0, yy2 - yy1 + 1)
#             inter = w * h
#             ovr = inter / (areas[i] + areas[order[1:]] - inter)

#             inds = np.where(ovr <= thresh)[0]
#             order = order[inds + 1]

#         return keep
    
#     def decode(self,loc, priors, variances):
#         """Decode locations from predictions using priors to undo
#         the encoding we did for offset regression at train time.
#         Args:
#             loc (tensor): location predictions for loc layers,
#                 Shape: [num_priors,4]
#             priors (tensor): Prior boxes in center-offset form.
#                 Shape: [num_priors,4].
#             variances: (list[float]) Variances of priorboxes
#         Return:
#             decoded bounding box predictions
#         """

#         boxes = torch.cat((
#             priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
#             priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
#         boxes[:, :2] -= boxes[:, 2:] / 2
#         boxes[:, 2:] += boxes[:, :2]
#         return boxes

#     def decode_landm(self,pre, priors, variances):
#         """Decode landm from predictions using priors to undo
#         the encoding we did for offset regression at train time.
#         Args:
#             pre (tensor): landm predictions for loc layers,
#                 Shape: [num_priors,10]
#             priors (tensor): Prior boxes in center-offset form.
#                 Shape: [num_priors,4].
#             variances: (list[float]) Variances of priorboxes
#         Return:
#             decoded landm predictions
#         """
#         landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
#                             priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
#                             priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
#                             priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
#                             priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
#                             ), dim=1)
#         return landms
    


    
#     # 변경 sy
#     # def run_inference(self, input_data_batch, ch_map_data_list=None):    
#     def run_inference(self, input_data_batch, unavailable_routing_data_batch, reference_CM=None):
#         if len(input_data_batch):
#             box_batch = list()
#             kps_batch = list()


#             parsed_input_batch = self.parse_input(input_data_batch)


#             frame_data = self.preprocess(parsed_input_batch)


#             output = self.inference(frame_data)

#             post_result = self.postprocess(output,parsed_input_batch)



#             output = self.parse_output(input_data_batch, post_result)


#         else:
#             output = []
#         return output, unavailable_routing_data_batch
    
# def module_load(logger):
#     retina = RETINAFACE(logger)
#     return retina    


# # ################### test ################### 
# # if __name__ == '__main__':
# #     import logging
# #     logger = logging.Logger('inference')
# #     torch.cuda.init() #파이프 라인 필수 

# #     ## 모델 생성
# # #     trt_engine_path = "/data/media_test/model_manager/engines/retina_fp16_2/retinanet2_fp16_016.trt"
# #     trt_engine_path = "/data/media_test/model_manager/engines/retinanet2_int8_5000_add_flip/retinanet2_int8_032.trt"
# #     retina = module_load(logger)
# #     retina.load(trt_engine_path)
    
# # #     샘플이미지 로드
# # #     img_path = "/data/media_test/model_manager/interfaces/Karina1.png"
# #     img_path = "/data/media_test/model_manager/test/test_image.jpeg"
# #     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# #     img = torch.from_numpy(img).to(torch.device("cuda"))
# #     img = img.permute(2, 0, 1) 
# #     print('img.shape',img.shape)

    
# # #     # empty인 경우 test 
# # #     img = torch.zeros(
# # #         [500, 3, 300], 
# # #         dtype=torch.float32,device=torch.device("cuda:0")
# # #     ).fill_(144)
    


# #     ## 더미데이터 생성
# #     input_data = dict()
# #     input_data["framedata"] = {"frame":img}
# #     input_data["bbox"] = [0,0,img.shape[2],img.shape[1]]
# #     input_data["scenario"] = "s"   
# #     input_data["data"] = None   
    
# #     ## 실제 데이터가 들어왔을때 배치만큼 리스트로 쌓여서 옴(4배치)
# #     input_data_batch = [input_data for i in range(16)]
# #     for i in range(3): 
# #         output = retina.run_inference(input_data_batch)
# #     s1 = time.time()
# #     for i in range(1000): 
# #         output = retina.run_inference(input_data_batch)
# #     s2 = time.time()
# #     print('average time : ', (s2-s1)/1000)
    
# #     output = retina.run_inference(input_data_batch)
# #     print(output)

