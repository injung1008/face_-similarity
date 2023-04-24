import tensorrt as trt
import numpy as np
import time
import torch
import os 


def restoreBboxScale(bbox,scale):
    x1,y1,x2,y2 = map(int, bbox)
    scale_w,scale_h = scale
    if scale_w == None or scale_h == None :
        return [x1,y1,x2,y2] 
    x1 = max(0,x1 / scale_w)
    y1 = max(0,y1 / scale_h)
    x2 = x2 / scale_w
    y2 = y2 / scale_h
    return [x1,y1,x2,y2] 



def getCropByFrame(frame,bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return frame[:,y1:y2,x1:x2]


class Engine:
    def __init__(self):
        self.out_shape = list()

        

    def load_engine(self, runtime,engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        return engine
 
    
    def make_context(self,trt_engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        self.engine = self.load_engine(runtime,trt_engine_path)

        self.context = self.engine.create_execution_context()

        self.input_shape = self.engine.get_profile_shape(0,0)[2]
        
        for i in range(self.engine.num_bindings) : 
            if not self.engine.binding_is_input(i) :
                self.out_shape.append(self.engine.get_binding_shape(i))
        


    
    def do_inference_v2(self, input_data):
        img_batch = input_data.shape[0]  

        self.input_shape[0] = img_batch
   
        for i in self.out_shape : 
            i[0] = img_batch

        output_datas = [torch.empty(size=tuple(i), dtype=torch.float32, device=torch.device("cuda")) for i in self.out_shape]


        self.context.set_binding_shape(0, tuple(self.input_shape))             

        bindings = None   
        
        bindings = [int(i.data_ptr()) for i in output_datas] 

        bindings.insert(0, int(input_data.data_ptr()))
        

        self.context.execute_async_v2(bindings,stream_handle=torch.cuda.current_stream().cuda_stream)
        

        return output_datas


