import numpy as np
import pickle
import os
import shutil

class VECTOR2PICKLE:
    def __init__(self, logger):
        self.flag = False
        self.batchsize = 16
        self.logger = logger
    def load(self, weights = None):
        self.flag = True
    def preprocess(self, x):
        return x
    def preprocess_for_calibration(self, x):
        return x
    def inference(self, x):
        return x
    def postprocess(self, x):
        return x
    

    def load_set_meta(self, channel_id, meta_data, channel_info_dict, model_name):
        
        save_path_dict = meta_data
        channel_info_dict[channel_id]['map_data'][model_name] = save_path_dict
        if not os.path.exists(save_path_dict):
            self.logger.error(f'{save_path_dict} Not exist')
            return 
        
        vector_gallery_path = save_path_dict.replace('gallery', 'vector')
        os.makedirs(vector_gallery_path, exist_ok=True)
        
        gallery_dir_list = [f for f in os.listdir(save_path_dict) if not f.startswith('.')] 
        
        for g_dir in gallery_dir_list:
            gallery_label_path = vector_gallery_path + '/' + g_dir
            if os.path.exists(gallery_label_path) :
                shutil.rmtree(gallery_label_path)
                os.mkdir(gallery_label_path)
            else :
                os.mkdir(gallery_label_path)
                


    def run_inference(self, queuedata_batch, unavailable_routing_data_batch, reference_CM=None):    
        count = 0
        for queuedata in queuedata_batch:
            count += 1
            framedata = queuedata['framedata']
            output = queuedata['data']
            channel_id = str(framedata['meta']['source']['channel_id'])
            img_path = framedata['path']
            

            label_id = img_path.split('/')[-2]

            img_path = img_path.replace('gallery', 'vector')
            img_path = os.path.splitext(img_path)
            
            save_path = f'{img_path[0]}.pickle'
            
            data = {
                str(label_id) : [output['label']]
                    }
            with open(save_path, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            queuedata['available'] = False
                
        return queuedata_batch, unavailable_routing_data_batch
    
    
def module_load(logger):
    return VECTOR2PICKLE(logger)