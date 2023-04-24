import numpy as np
import os
import pickle

class SIMILARITY:
    def __init__(self, logger):
        self.flag = False
        self.batchsize = 16
        self.score_thresh = 0.3
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
    
        #########################
    # * 전달받은 피클 데이터 리스트를 로드하여 딕셔너리로 반환
    # * Input :
    # *    data_list : 피클 데이터 리스트
    # * Output :
    # *    data_dict : 로드된 피틀 데이터 딕셔너리
    # vector pickle load
    def load_set_meta(self, channel_id, meta_data, channel_info_dict, model_name):
        pickle_path = meta_data
        
        
        total_pickle_list = list()
        vector_dict = dict()
        vector_dict['label_list'] = []
        vector_dict['label_vectors'] = None
        try : 
            if os.path.exists(pickle_path) == True : 
                pickle_dir_list = [f for f in os.listdir(pickle_path) if not f.startswith('.')]

                for label_dir in pickle_dir_list:
                    pickle_label_path = os.path.join(pickle_path, label_dir)
                    if not os.path.isdir(pickle_label_path) :
                        continue
                    fileEx = r'.pickle'
                    pickle_list = [file for file in os.listdir(pickle_label_path) if file.endswith(fileEx)]

                    for pickle_file in pickle_list : 
                        pickle_file_path = os.path.join(pickle_label_path,pickle_file)
                        total_pickle_list.append(pickle_file_path)
                vector_data_dict = self.load_vector_pickle(total_pickle_list)

                if pickle_path == '/xaiva/nfs/star' : 
                    vector_dict['label_list'] = [ label for label, _ in vector_data_dict.items()]
                    vector_dict['label_vectors'] = np.array([ vectors for _, vectors in vector_data_dict.items()])

                else : 
                    total_mask_np , vector_data_dict = self.make_mask_for_custom(vector_data_dict)

                    vector_dict['label_list'] = [ label for label, _ in vector_data_dict.items()]
                    vector_dict['label_vectors'] = np.array([ vectors for _, vectors in vector_data_dict.items()])
                    vector_dict['mask_np'] = total_mask_np
                    self.logger.info(f'total_mask_np : {total_mask_np}')
                    self.logger.info(f'label_list : {vector_dict["label_list"]}')
        
        except Exception as e : 
            self.logger.error(f'pickle_path ERROR : {pickle_path}')

        channel_info_dict[channel_id]['map_data'][model_name] = vector_dict
        self.logger.info(f'label : {vector_dict["label_list"]}')

    def make_mask_for_custom(self, vector_data_dict):
        max_cnt = 0
        for _, label_vectors in vector_data_dict.items() :
            if max_cnt > len(label_vectors) : 
                pass 
            else :
                max_cnt = len(label_vectors)
        
        total_mask = []
        for label, label_vectors in vector_data_dict.items() :
            mask_list = [False for _ in range(len(label_vectors))]
            for i in range(max_cnt-len(label_vectors)) :
                mask_list.append(True)

            total_mask.append(mask_list)
        
        total_mask_np = np.array(total_mask)

        for label, label_vectors in vector_data_dict.items() :
            m_img = np.zeros(512,)

            len_label = len(label_vectors)
            for i in range(max_cnt-len_label) :
                label_vectors.append(m_img)


        return total_mask_np, vector_data_dict
        
    
    def load_vector_pickle(self, data_list):
        data_dict = dict()
        for data_path in data_list:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                
            for k, vs in data.items():
                if k not in data_dict.keys():
                    data_dict[k] = []
                for v in vs:
                    data_dict[k].append(v)
        
        return data_dict   
    
    
    
    def parse_input(self,input_data_batch):
        res = []
        for input_data in input_data_batch:

            vector = input_data['data']['label']
            channel_id = input_data['framedata']['meta']['source']['channel_id']
            retina_score = input_data['data']['score']
            res.append([vector, channel_id,retina_score])
        return res 
    
    def parse_output(self,input_data_batch,output_batch):
        res = []
   
        for idx_i, (data, (label, scores)) in enumerate(zip(input_data_batch, output_batch)): 
            framedata = data['framedata']
            scenario = data['scenario']
            bbox = data['bbox']
            
            input_data = dict()
            input_data["framedata"] = framedata
            input_data["bbox"] = bbox
            input_data["scenario"] = scenario   
            input_data["data"] = {"label":label, "score":scores}
            input_data["available"] = True
            res.append(input_data)
        return res
    
    
    #def run_inference(self, input_data_batch, ch_map_data_list=None):    
    # 변경 sy
    def run_inference(self, input_data_batch, unavailable_routing_data_batch, reference_CM=None):    
        if len(input_data_batch):
            new_datas = []

            parsed_input_batch = self.parse_input(input_data_batch)

            result_batch = []
            for targets, channel_id,retina_score in parsed_input_batch:

                label_list = reference_CM.channel_info_dict[channel_id]['map_data']['similarity_None']['label_list']
                label_vectors = reference_CM.channel_info_dict[channel_id]['map_data']['similarity_None']['label_vectors']

                # 비교 벡터가 없을 경우 
                if not len(label_list):
                    label = '81'
                    best_score = 0.3
                    result_batch.append([label, retina_score])

                    continue

                best_s_label = '81'
                best_score = retina_score
                
                targets = np.array(targets)
                scores = label_vectors.dot(targets)/(np.linalg.norm(label_vectors, axis=2) * np.linalg.norm(targets))
                if 'mask_np' not in reference_CM.channel_info_dict[channel_id]['map_data']['similarity_None'] : 
                    result = np.mean(scores,axis=1)
                    b_score_idx = np.argmax(result)
                    best_s_label = label_list[b_score_idx]
                    best_score = result[b_score_idx]
                else : 
                    total_mask_np = reference_CM.channel_info_dict[channel_id]['map_data']['similarity_None']['mask_np']
                    masked_score = np.ma.array(scores, mask=total_mask_np)
                    result = np.mean(masked_score,axis=1)
                    b_score_idx = np.argmax(result)
                    best_s_label = label_list[b_score_idx]
                    best_score = result[b_score_idx]
                

                ### self.score_thresh = 0.3 ###
                if best_score > self.score_thresh:

                    result_batch.append([best_s_label, best_score])
                else:
                    # Unknown label id
                    label = '81'
                    result_batch.append([label, retina_score])
                

                

            output = self.parse_output(input_data_batch,result_batch)
        else:
            output = []
        return output, unavailable_routing_data_batch
    

def module_load(logger):
    similarity = SIMILARITY(logger)
    return similarity



# if __name__ == '__main__':
#     import logging
#     logger = logging.Logger('inference')
#     vector = 0
#     similarity = module_load(logger)


#     ## 더미데이터 생성
#     input_data = dict()
#     input_data["framedata"] = {"frame":None}
#     input_data["bbox"] = [None]
#     input_data["scenario"] = "s"   
#     input_data["data"] = vector
    
#     ## 실제 데이터가 들어왔을때 배치만큼 리스트로 쌓여서 옴(4배치)
#     input_data_batch = [input_data for i in range(4)]

#     output = similarity.run_inference(input_data_batch)
#     print(output)
#####################################################################################################

# import numpy as np
# import os
# import pickle

# class SIMILARITY:
#     def __init__(self, logger):
#         self.flag = False
#         self.batchsize = 16
#         self.score_thresh = 0.3
#         self.logger = logger
#     def load(self, weights = None):
#         self.flag = True
#     def preprocess(self, x):
#         return x
#     def preprocess_for_calibration(self, x):
#         return x
#     def inference(self, x):
#         return x
#     def postprocess(self, x):
#         return x
    
#         #########################
#     # * 전달받은 피클 데이터 리스트를 로드하여 딕셔너리로 반환
#     # * Input :
#     # *    data_list : 피클 데이터 리스트
#     # * Output :
#     # *    data_dict : 로드된 피틀 데이터 딕셔너리
#     # vector pickle load
#     def load_set_meta(self, channel_id, meta_data, channel_info_dict, model_name):
#         pickle_path = meta_data
#         total_pickle_list = list()
#         vector_dict = dict()
#         if os.path.exists(pickle_path) == True : 
#             pickle_dir_list = [f for f in os.listdir(pickle_path) if not f.startswith('.')]

#             for label_dir in pickle_dir_list:
#                 pickle_label_path = os.path.join(pickle_path, label_dir)
#                 fileEx = r'.pickle'
#                 pickle_list = [file for file in os.listdir(pickle_label_path) if file.endswith(fileEx)]

#                 for pickle_file in pickle_list : 
#                     pickle_file_path = os.path.join(pickle_label_path,pickle_file)
#                     total_pickle_list.append(pickle_file_path)

#             vector_dict = self.load_vector_pickle(total_pickle_list)

#         channel_info_dict[channel_id]['map_data'][model_name] = vector_dict
# #         self.logger.info(f'vector_dict : {vector_dict.keys()}')

        
    
#     def load_vector_pickle(self, data_list):
#         data_dict = dict()
#         for data_path in data_list:
#             with open(data_path, 'rb') as f:
#                 data = pickle.load(f)
                
#             for k, vs in data.items():
#                 if k not in data_dict.keys():
#                     data_dict[k] = []
#                 for v in vs:
#                     data_dict[k].append(v)
#         return data_dict   
    
    
    
#     def parse_input(self,input_data_batch):
#         res = []
#         for input_data in input_data_batch:

#             vector = input_data['data']['label']
#             channel_id = input_data['framedata']['meta']['source']['channel_id']
#             retina_score = input_data['data']['score']
#             res.append([vector, channel_id,retina_score])
#         return res 
    
#     def parse_output(self,input_data_batch,output_batch):
#         res = []
   
#         for idx_i, (data, (label, scores)) in enumerate(zip(input_data_batch, output_batch)): 
#             framedata = data['framedata']
#             scenario = data['scenario']
#             bbox = data['bbox']
            
#             input_data = dict()
#             input_data["framedata"] = framedata
#             input_data["bbox"] = bbox
#             input_data["scenario"] = scenario   
#             input_data["data"] = {"label":label, "score":scores}
#             input_data["available"] = True
#             res.append(input_data)
#         return res
    
    
#     #def run_inference(self, input_data_batch, ch_map_data_list=None):    
#     # 변경 sy
#     def run_inference(self, input_data_batch, unavailable_routing_data_batch, reference_CM=None):    
#         if len(input_data_batch):
#             new_datas = []

#             parsed_input_batch = self.parse_input(input_data_batch)

#             result_batch = []
#             for targets, channel_id,retina_score in parsed_input_batch:
     
#                 labels = reference_CM.channel_info_dict[channel_id]['map_data']['similarity_None']

#                 comp_list = []
#                 label_list = []
                
#                 for label_id, label_vectors in labels.items():
#                     targets = np.array(targets)
#                     label_vectors = np.array(label_vectors)
#                     scores = label_vectors.dot(targets)/(np.linalg.norm(label_vectors, axis=1) * np.linalg.norm(targets))
#                     result = np.mean(scores)
#                     label_list.append(label_id)
#                     comp_list.append(result)


#                 # 비교 벡터가 없을 경우 
#                 if not len(comp_list):
#                     label = '81'
#                     best_score = 0.3
#                     result_batch.append([label, retina_score])

#                     continue

                
#                 best_score = max(comp_list)
#                 best_idx = comp_list.index(best_score)


#                 ### self.score_thresh = 0.3 ###
#                 if best_score > self.score_thresh:
#                     label = label_list[best_idx]
#                     result_batch.append([label, best_score])
#                 else:
#                     # Unknown label id
#                     label = '81'
#                     result_batch.append([label, retina_score])
                
                

#             output = self.parse_output(input_data_batch,result_batch)
#         else:
#             output = []
#         return output, unavailable_routing_data_batch
    

# def module_load(logger):
#     similarity = SIMILARITY(logger)
#     return similarity

# # if __name__ == '__main__':
# #     import logging
# #     logger = logging.Logger('inference')
# #     vector = 0
# #     similarity = module_load(logger)


# #     ## 더미데이터 생성
# #     input_data = dict()
# #     input_data["framedata"] = {"frame":None}
# #     input_data["bbox"] = [None]
# #     input_data["scenario"] = "s"   
# #     input_data["data"] = vector
    
# #     ## 실제 데이터가 들어왔을때 배치만큼 리스트로 쌓여서 옴(4배치)
# #     input_data_batch = [input_data for i in range(4)]

# #     output = similarity.run_inference(input_data_batch)
# #     print(output)