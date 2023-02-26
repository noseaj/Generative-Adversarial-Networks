import glob
import json
import cv2
import os
import albumentations
from PIL import Image

class Preprocessing():
    def __init__(self):
        self.file_path = "data"
      
    def leak_extract(self):
        os.mkdir(self.file_path+'\Leak')
        L = os.listdir(self.file_path) # [L_01-1,L_01-2,L_01-3,L_01-4,L_01-5,L_01-6]
        for idx, name in enumerate(L):
            if os.path.isdir(self.file_path+ '\\' + name) == True:
                R_T = os.listdir(self.file_path+'\\' + name) # [Rotate,Top]
                for idx2, name2 in enumerate(R_T):
                    if os.path.isdir(self.file_path+'\\'+name+'\\'+name2) == True:
                        img_path = self.file_path+'\\'+name+'\\'+name2
                        
                        img_file_list = glob.glob(img_path + '\\*.jpg')
                        
                        json_file_list = glob.glob(img_path + '\\*.json')
                        
                        for idx3, name3 in enumerate(img_file_list):
                            match_name = name3.split('\\')[-1].split('.')[0]
                            img = cv2.imread(f'{img_path}/{match_name}.jpg', cv2.IMREAD_COLOR)
                            if (img_path + '\\' +  match_name + '.json') in json_file_list:
                                with open(f'{img_path}\{match_name}.json') as f:
                                    json_object = json.load(f)
                                    
                                    for j in range(len(json_object['shapes'])):
                                        leak1 = json_object['shapes'][j]['points']
                                        if (len(leak1) > 4): # 리크가 점 1개면 오류 발생
                                            min_x = leak1[0][0]
                                            max_x = leak1[0][0]
                                            min_y = leak1[0][1] 
                                            max_y = leak1[0][1]

                                            for k in range(len(leak1)):
                                                #min_x 찾기
                                                if leak1[k][0] < min_x:
                                                    min_x = leak1[k][0]
                                                else:
                                                    pass
                                                #max_x 찾기
                                                if leak1[k][0] > max_x:
                                                    max_x = leak1[k][0]
                                                else:
                                                    pass
                                                #min_y 찾기
                                                if leak1[k][1] < min_y:
                                                    min_y = leak1[k][1]
                                                else:
                                                    pass
                                                #max_y 찾기
                                                if leak1[k][1] > max_y:
                                                    max_y = leak1[k][1]
                                                else:
                                                    pass

                                            x=int(min_x)
                                            y=int(min_y)
                                            w=int(max_x - min_x)
                                            h=int(max_y - min_y)
                                            if(w==0 or h ==0): #일자 리크는 오류
                                                break
                                            roi = img[y:y+h, x:x+w]
                                            save_path = f'{self.file_path}\Leak\{match_name}_{j+1}.jpg'
                                            cv2.imwrite(save_path, roi)
                                        else:
                                            pass
        
    def remove_grid_line(self):
        new_file_path = self.file_path+'\Leak\\'
        imglist = glob.glob(new_file_path + '\*.jpg')
        temp = set()
        for img_name in imglist:
            leak = Image.open(img_name)
            for i in range(0, leak.size[1]):
                for j in range(0, leak.size[0]):
                    r,g,b = leak.getpixel((j,i))
                    if ((127 <= r <= 255) and (127 <= g <= 255) and (0<= b <= 100)):
                        temp.add(img_name)
        grid_leak_list = [x for x in temp]
        no_grid_leak_list = list(set(imglist) - set(grid_leak_list))

        os.mkdir(self.file_path + "\Leak_file_path")
        for leak_name in no_grid_leak_list:
            leak = Image.open(leak_name)
            leak_name = leak_name.split('\\')[-1]
            leak.save(self.file_path + '\Leak_file_path\\'+ leak_name)
            leak.close()

        leak_file_path = self.file_path + '\Leak_file_path'
        leak_file_list = os.listdir(leak_file_path)
        return leak_file_path , leak_file_list
    

    def resize(self, leak_file_path, leak_file_list):
        os.mkdir(leak_file_path + "\Resize")
        for img_name in leak_file_list:
            img = Image.open(leak_file_path + '\\' + img_name)
            resize_img = img.resize((64, 64))
            resize_img.save(leak_file_path + '\Resize\\' + img_name)
            img.close()

        resize_file_path = leak_file_path + '\Resize'
        resize_file_list = os.listdir(resize_file_path)
        return resize_file_path, resize_file_list

    def augmentation(self, resize_file_path, resize_file_list):
        transform_1 = albumentations.HorizontalFlip(p=1)
        transform_2 = albumentations.RandomRotate90(p=1)
        transform_3 = albumentations.VerticalFlip(p=1)

        os.mkdir(resize_file_path + "\Leak_for_Modeling")
        for img_name in resize_file_list:
            image = cv2.imread(resize_file_path + '\\'+ img_name)

            augmentations_1 = transform_1(image=image)
            augmentation_img_1 = augmentations_1['image']
            cv2.imwrite(resize_file_path + '\Leak_for_Modeling\\'+'h_flip'+img_name, augmentation_img_1)

            augmentations_2 = transform_2(image=image)
            augmentation_img_2 = augmentations_2['image']
            cv2.imwrite(resize_file_path + '\Leak_for_Modeling\\'+'r_rotate90'+img_name, augmentation_img_2)

            augmentations_3 = transform_3(image=image)
            augmentation_img_3 = augmentations_3['image']
            cv2.imwrite(resize_file_path + '\Leak_for_Modeling\\'+'v_flip'+img_name, augmentation_img_3)

            modeling_file_path = resize_file_path + '\Leak_for_Modeling'
        return modeling_file_path