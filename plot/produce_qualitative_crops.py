import cv2
import os
import os.path
import shutil
import glob

EXPERIMENTS_PATH="/trinity/home/mangelini/data/mangelini/Pruning/Test/finetuned(200)"
ORIGINAL_IMAGES_PATH="/trinity/home/mangelini/data/mangelini/Pruning/Dataset/eval/benchmark"
OUT_PATH="/trinity/home/mangelini/develop/drln_pruning/plot/Images"
CROP_SIZE_X = 3*10
CROP_SIZE_Y = 4*10
# Name, size, crop, scale
IMAGES = [
    ("['Manga109']_['HanzaiKousyouninMinegishiEitaroux2']", (824,1168), (725,925), 2),
    ("['Manga109']_['HanzaiKousyouninMinegishiEitaroux4']", (824,1168), (725,925), 4),
    
    #("['Urban100']_['img031x2']", (1024,680), (450,200), 2),
    #("['Urban100']_['img031x4']", (1024,680), (250,200), 4),
    
    ("['Urban100']_['img049x2']", (1024,680), (250, 480), 2),
    ("['Urban100']_['img049x4']", (1024,680), (250, 480), 4),
    
    ("['B100']_['253027x2']", (400,320), (200, 150), 2),
    ("['B100']_['253027x4']", (400,320), (200, 150) ,4),
    
    #("['B100']_['159008x2']", (480,320), (300, 150), 2),
    #("['B100']_['159008x4']", (480,320), (300, 150) ,4),
    
]


METHODS = {
    "growing_reg": "GReg",
    "group_norm": "GLasso",
    "norm": "Lasso"
}

def getImageInfo(image_name: str):
    split_name = image_name.replace("['", "").replace("']", "").replace("x2", "").replace("x4", "").split("_")
    return split_name[0], split_name[1]

def returnDictMatch(key_check, d: dict):
    for k in d.keys():
        if k in key_check: return d[k]
    
    raise Exception ("Key Not Found!!!")

def extractExperimentInfo(exp):
    model = "DRLN" if "original" in exp else "SWIN"
    method = returnDictMatch(exp, METHODS)
    return (model, method)
    

if __name__ == '__main__':
    if os.path.exists(OUT_PATH):shutil.rmtree(OUT_PATH)
    experiments = glob.glob(f'{EXPERIMENTS_PATH}/*' )

    # Filter No Retrain
    experiments = [r for r in experiments if 'DIV2K-50' in r]
    # Only Upsaple Pruning
    experiments = [r for r in experiments if 'False,True))' in r]
    # Filter Only 8 steps
    experiments = [r for r in experiments if '(8,' in r]
    #Remove Random
    experiments = [r for r in experiments if 'random' not in r]

    for img_name, size, crop, scaling in IMAGES:
        img_out= f'{OUT_PATH}/{img_name}_x{scaling}'
        os.makedirs(img_out, exist_ok=True)
        orignial_path = f'{img_out}/original.png'
        original_crop_path = f'{img_out}/original_crop.png'
        
        # Extract Crop Pos
        crop_pos_y, crop_pos_x = crop
        
        # Process Original
        sequence, name = getImageInfo(img_name)
        cv2_orignal = cv2.imread(f"{ORIGINAL_IMAGES_PATH}/{sequence}/HR/{name}.png")
        
        cv2_original_crop = cv2_orignal[ crop_pos_x:crop_pos_x+CROP_SIZE_X, crop_pos_y:crop_pos_y+CROP_SIZE_Y]
        cv2.imwrite(original_crop_path, cv2_original_crop)
        
        cv2.rectangle(
                cv2_orignal, 
                #(crop_pos_x, crop_pos_y), (crop_pos_x+CROP_SIZE_X, crop_pos_y+CROP_SIZE_Y), 
                (crop_pos_y, crop_pos_x), (crop_pos_y+CROP_SIZE_Y, crop_pos_x+CROP_SIZE_X,), 
                (0,0,255), 2
        ) 
        cv2.imwrite(orignial_path, cv2_orignal)
        
        for e in experiments:
            if f"x{scaling}," not in e: continue
            if not os.path.exists(f"{e}/Infer"): continue
            if not os.path.exists(f"{e}/Infer/unpruned"): continue
            if not os.path.exists(f"{e}/Infer/8"): continue

            model, method = extractExperimentInfo(e)
            unpruned_crop_path = f'{img_out}/{model}_unpruned_crop.png'
            pruned_crop_path = f'{img_out}/{model}_{method}_crop.png'
            
            if not os.path.exists(unpruned_crop_path):
                image_path_unpruned = f"{e}/Infer/unpruned/{img_name}.png"
                cv2_unpruned = cv2.imread(image_path_unpruned)
                cv2_unpruned_crop = cv2_unpruned[crop_pos_x:crop_pos_x+CROP_SIZE_X, crop_pos_y:crop_pos_y+CROP_SIZE_Y]
                cv2.imwrite(unpruned_crop_path, cv2_unpruned_crop)
            
            
            image_path_pruned = f"{e}/Infer/8/{img_name}.png"
            cv2_pruned = cv2.imread(image_path_pruned)
            cv2_pruned_crop = cv2_unpruned[crop_pos_x:crop_pos_x+CROP_SIZE_X, crop_pos_y:crop_pos_y+CROP_SIZE_Y]
            cv2.imwrite(pruned_crop_path, cv2_pruned_crop)
        
        