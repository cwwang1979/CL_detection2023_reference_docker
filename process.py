
import SimpleITK
from pathlib import Path
import numpy as np
from pandas import DataFrame
import torch
import json
import os
from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import pandas as pd
import mmdet
from mmdet.apis import inference_detector,init_detector
import pickle,mmcv

class Cldetection_alg_2023(DetectionAlgorithm):
    def __init__(self,input_path = Path("/input/images/lateral-dental-x-rays/"),
            output_file = Path("/output/orthodontic-landmarks.json")):
        self._input_path = input_path
        self._output_file = output_file
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),                
            

        )
        # TODO: use GPU if available, otherwise use the CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fin_pred=pd.DataFrame()

        # TODO: You may adapt or propose your AI model/algorithm with other toolbox or design your own the AI model.

        # TODO: This part should be your AI model weights
        self.model = pickle.load(open("/opt/algorithm/mdl.pkl",'rb'))
        self.cfg = pickle.load(open("/opt/algorithm/cfg.pkl",'rb'))
        self.model.cfg = self.cfg

        print("==> Using ", self.device)
        print("==> Initializing model")
        print("==> Weights loaded")

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)
        self.idx=idx
        
        # candidates scoring
        candidates = self.predict(input_image=input_image)

        # Write resulting candidates to result.json for this case
        return dict(type="Multiple points", points=candidates, version={ "major": 1, "minor": 0 })


    def predict(self, *, input_image: SimpleITK.Image) -> DataFrame:
        
        # Extract a numpy array with image data from the SimpleITK Image
        image_data = SimpleITK.GetArrayFromImage(input_image)
        fin_pred=pd.DataFrame()

        for id in range(image_data.shape[0]):
            
            # Load individual image from image stack
            image = np.array(image_data[id,:,:,:])

            # Inferencing process
            img = mmcv.imread(image)
            results = inference_detector(self.model, img)
            res=np.array(results,dtype=object)

            np.set_printoptions(suppress=True, precision=2)

            #converting from bbox to single point coordinate    
            temp=[]
            for x in range(len(res)):
                ress=res[x]
                resssize=ress.size//5
                
                if resssize==0:
                    #generate null prediction array if there is no prediction 
                    ress2=[0,0,0,0,0]
                    ress3=[0,0,0,x+1]
                
                else:
                    ress2=ress[0]
                    #back projection from bbox (x_1, y_1, x_2, y_2, probability, name(class)) to single point prediction
                    ress3=[ress2[0]+((ress2[2]-ress2[0])//2),ress2[1]+((ress2[3]-ress2[1])//2),ress2[4],x+1]

                temp.append(ress3)
            
            # Converting prediction result to [Grand-challenge.org] DL-detection .json format
            tempp=np.array(temp)  

            dftemp= pd.DataFrame(tempp,columns=['x','y','probability','name']).round(2)
            dftemp['name']=dftemp['name'].values.astype(int).astype(str)
            dftemp[['x','y']]=dftemp[['x','y']].values.astype(int)

            dftemp=dftemp[['name','x','y','probability']]
            
            off=self.idx*50

            img_id=id+1+off

            imgl=np.full(38,img_id)
            imgn=pd.DataFrame(imgl, columns = ['z'])

            merge=pd.concat([imgn,dftemp], axis=1, join='outer')
            merge=merge[['name','x','y','z','probability']]
        
            fin_pred=fin_pred.append(merge, ignore_index=True)

        self.fin_pred=self.fin_pred.append(fin_pred, ignore_index=True)
        fn=self.fin_pred.values.tolist()

        result = [{"name": c[0],"point": c[1:4]} for c in fn]

        nres= dict(name="Orthodontic landmarks",type="Multiple points",
                points=result, version={ "major": 1, "minor": 0 })
        
        p_res=json.dumps(nres,indent=4)
        print(p_res)
        
        self.final_result=p_res
        res = isinstance(self.final_result, str)
        if res==True:
            print("======")
            print('The prediction is succesfully generated!')
            print("======")
        else:
            print("======")
            print('NO prediction is generated!')
            print("======")

        return DataFrame(result)

if __name__ == "__main__":
    Cldetection_alg_2023().process()
    
