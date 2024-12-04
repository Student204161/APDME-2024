import sys
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.dataset_readers import sceneLoadTypeCallbacks
from scipy.spatial.transform import Rotation as Rot
import numpy as np

import pickle

import cv2

import sys

#Variables:
object_name=sys.argv[1]
thresh_coef=float(sys.argv[2])
round_num=sys.argv[3]
dil_size=int(sys.argv[4]) #25
erode_size=int(sys.argv[5]) #20
model_name=sys.argv[6]
reproj_masks=sys.argv[7]
num_iterations=sys.argv[8]
rmv_sz=float(sys.argv[9])
rmv_dist=float(sys.argv[10])
white_background=sys.argv[11]
# Set up command line argument parser
parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")


# Add the following line to parse known arguments and ignore the rest
args, unknown = parser.parse_known_args()
args.model_path = rf"/dtu/blackhole/07/155527/camproj/data/GS_models/{model_name}/{object_name}/{num_iterations}_round_{round_num}"
args.iteration = num_iterations
args.eval = True
args.resolution=1
args.source_path=rf'/dtu/blackhole/07/155527/camproj/data/colmap_info/{object_name}'
args.quiet=False
args.data_device='cuda'
args.max_sh_degree=3
print("Rendering " + args.model_path)


def render_all_test_cameras(scene, pipeline, background, path_to_save,full_img_dir, undistorted_mask_path, scene_info,train_split):
    if not os.path.exists(path_to_save):
        os.path.makedirs(path_to_save)
    
    if train_split:
        test_or_train = scene.getTrainCameras()
    else:
        test_or_train = scene.getTestCameras()
    
    for i, cam in enumerate(test_or_train):
        
        if train_split:
            full_img = cv2.imread(os.path.join(full_img_dir, scene_info.train_cameras[i].image_name + '.jpg'))
            img_mask = cv2.imread(os.path.join(undistorted_mask_path, scene_info.train_cameras[i].image_name + '.png'),cv2.IMREAD_GRAYSCALE)
        else:
            full_img = cv2.imread(os.path.join(full_img_dir, scene_info.test_cameras[i].image_name + '.jpg'))
            img_mask = cv2.imread(os.path.join(undistorted_mask_path, scene_info.test_cameras[i].image_name + '.png'),cv2.IMREAD_GRAYSCALE)


        img = render(cam, scene.gaussians, pipeline.extract(args), background)["render"]
        img = img.permute(1,2,0)
        img = img.detach().cpu().numpy()
        img = img - img.min()
        img = img / img.max()
        # img = img * 255
        # img = img.astype('uint8')
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img * 255
        img = img.astype('uint8')
        result = (img_gray > 0).astype('int')
        result = (result*255).astype('uint8')
        dil_result = cv2.dilate(result, np.ones((dil_size, dil_size),np.uint8), iterations = 1)
        erode_result = cv2.erode(result, np.ones((erode_size, erode_size),np.uint8), iterations = 1)
                    
        #area that is in dil_result but not in erode_result = dil_result - erode_result
        #unchanged_area = cv2.bitwise_and(dil_result, cv2.bitwise_not(erode_result))
        
        # plt.imshow(plo)
        # plt.show()
        
        result1 = cv2.bitwise_and(img_mask,dil_result)
        # result = cv2.bitwise_or(result1, erode_result)

        # plt.imshow(img_mask)
        # plt.show()
        # plt.imshow(result)
        # plt.show()

        # result = (density > 0).astype('int')
        #result = (result*255).astype('uint8')

        result = (result.astype('float32'))/255
        for channel in range(3):
            full_img[:,:,channel] = full_img[:,:,channel]*result
        if train_split:
            full_img = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path_to_save + scene_info.train_cameras[i].image_name + '.jpg',full_img)
            #print('image saved to: ' + scene_info.train_cameras[i].image_name + '.jpg')
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path_to_save + scene_info.test_cameras[i].image_name + '.jpg',img)
            #print('image saved to: ' + scene_info.test_cameras[i].image_name + '.jpg')


def filter_gaussians_by_list_and_save(scene, pipeline, background,filter_name, full_img_dir,undistorted_mask_path, scene_info, vertices_keep_indx=None,color_red=False,only_color=False,save=True):
    ## base mvg - new masks
    #if path to save does not exist, create it

    if vertices_keep_indx and not only_color:
        scene.gaussians._xyz = scene.gaussians._xyz[vertices_keep_indx]
        scene.gaussians._scaling = scene.gaussians._scaling[vertices_keep_indx]
        scene.gaussians._rotation = scene.gaussians._rotation[vertices_keep_indx]
        scene.gaussians._features_dc = scene.gaussians._features_dc[vertices_keep_indx]
        scene.gaussians._features_rest = scene.gaussians._features_rest[vertices_keep_indx]
        scene.gaussians._opacity = scene.gaussians._opacity[vertices_keep_indx]

    if color_red:
        scene.gaussians._features_dc = torch.ones_like(scene.gaussians._features_dc) * torch.tensor([1,0,0], device="cuda")
    
    if only_color and vertices_keep_indx:
        scene.gaussians._features_dc[vertices_keep_indx] = torch.ones_like(scene.gaussians._features_dc[vertices_keep_indx]) * torch.tensor([1,0,0], device="cuda")

    path_to_save = rf'/dtu/blackhole/07/155527/camproj/data/undistorted_images/{filter_name}/{object_name}/{round_num}/'

    if not os.path.exists(path_to_save):
        #create the directory
        os.makedirs(path_to_save)

    render_all_test_cameras(scene, pipeline, background, path_to_save, full_img_dir,undistorted_mask_path, scene_info,True)

    ## base mvg - test renders of GS
    path_to_save = rf'/dtu/blackhole/07/155527/camproj/data/GS_models/{filter_name}/test_{object_name}/{round_num}/'
    #if path to save does not exist, create it
    if not os.path.exists(path_to_save):
        #create the directory
        os.makedirs(path_to_save)
    if save:
        render_all_test_cameras(scene, pipeline, background, path_to_save, full_img_dir,undistorted_mask_path, scene_info,False)

    return scene



scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)

# Initialize system state (RNG)
def load_3DGS(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        return scene

safe_state(args.quiet)
scene = load_3DGS(model.extract(args), args.iteration, pipeline.extract(args))

# bg_color = [0,0,0] #[1,1,1] if dataset.white_background else [0, 0, 0]
# background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

# img = render(scene.getTestCameras()[0], scene.gaussians, pipeline.extract(args), background)["render"]

bg_color = [1,1,1] if white_background else [0, 0, 0]
white_str = '_white' if white_background else ''

background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

img = render(scene.getTestCameras()[0], scene.gaussians, pipeline.extract(args), background)["render"]

#display 3xHxW image
img = img.permute(1,2,0)
img = img.detach().cpu().numpy()
img = img - img.min()
img = img / img.max()
img = img * 255
img = img.astype('uint8')
import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.show()
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite('bl_okay_im.jpg',img)

# scene.gaussians

from collections import Counter
import numpy as np
import json
def GS_reproj(cam_json_path, path_to_masks, scene):
    with open(cam_json_path) as f:
        cam_json = json.load(f)
 
    test_imgs_num = 0
    for i in range(len(cam_json)):
        if i % 8 == 0:
            test_imgs_num +=1

    cam_json = cam_json[test_imgs_num:]
    list3d = scene.gaussians.get_xyz.cpu().detach().numpy()
    c = 0
    dist_coefs = None #np.array([k1,k2,p1,p2,k3]); dist_coefs.shape = (5,1)
    tqdm_bar = tqdm(cam_json) #tqdm(colmap_transforms_json['frames'])
    inside_mask_dict = {tuple(point): 0 for point in list3d}
    inside_image_list = np.zeros(len(list3d))
    counter = Counter(inside_mask_dict)
    
    for im_data in tqdm_bar:
        c += 1

        int_matrix = np.array([[im_data['fx'], 0, im_data['width']/2],
                [0, im_data['fy'], im_data['height']/2],
                [0, 0, 1]])

        R_gaussian = np.asarray(im_data['rotation'])
        T_gaussian = np.asarray(im_data['position'])
        t_mat = np.hstack((R_gaussian, T_gaussian.reshape(-1, 1)))
        transf_matrix = np.vstack((t_mat,[0,0,0,1]))
        transf_matrix_inv = np.linalg.inv(transf_matrix)

        R = transf_matrix_inv[:3,:3]
        t = transf_matrix_inv[:3,3]

        rvec, _ = cv2.Rodrigues(R)
        cam_pos = t
        
        path_mask = path_to_masks + im_data['img_name'] + '.png' #im_data['file_path'].split('/')[-1].replace('.jpg', '.png') 
        img_mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE) # 
        # kernel = np.ones((9, 9), np.uint8) 
        # img_mask = cv2.dilate(img_mask, kernel, iterations=1) 
        #img_mask = cv2.undistort(img_mask,int_matrix,dist_coefs)
        
        h, w = img_mask.shape
        fast_uv = cv2.projectPoints(np.asarray(list3d), rvec, t, int_matrix, dist_coefs)

        fast_uv_rounded = np.asarray(np.round(fast_uv[0].reshape(len(fast_uv[0]),2)),dtype='int')

        inside_img = np.all([fast_uv_rounded[:,0] >= 0,fast_uv_rounded[:,0] < w,0 <= fast_uv_rounded[:,1],fast_uv_rounded[:,1] < h], axis=0)
        fast_uv_rounded_copy = np.copy(fast_uv_rounded)
        fast_uv_rounded[~inside_img] = 0
        inside_image_list += inside_img

        inside_mask = np.full(len(fast_uv_rounded),False)
        inside_mask = img_mask[fast_uv_rounded[:,1],fast_uv_rounded[:,0]] > 0
        inside_mask[~inside_img] = False
        inside_mask_xyz = list3d[inside_mask]

        inside_mask_xyz = tuple(map(tuple, inside_mask_xyz))
        counter.update(inside_mask_xyz)

        # Visualize the mask and 
        # inside_mask_uv = fast_uv_rounded_copy[inside_mask]
        # new_img_mask = np.zeros((h,w))
        # for uv in inside_mask_uv:
        #     new_img_mask[uv[1],uv[0]] = 1
        
        # plt.figure(figsize=(20,20))
        # plt.imshow(new_img_mask[:,:])
        # plt.imshow(img_mask[:,:], cmap='jet', alpha=0.5)
        # plt.show()
        
    inside_mask_dict = dict(counter)

    inside_img_dict = {}
    for x in range(len(list3d)):
        inside_img_dict[tuple(list3d[x])] = inside_image_list[x]
    
    return inside_mask_dict, inside_img_dict

    
import sys
def filter_fun_4_mesh(inside_mask_dict, inside_img_dict, list3d, thresh_coefs,tot_frames_for_no_norm=0):
    #assert len(thresh_coefs) == 2
    if tot_frames_for_no_norm > 0:
        keep_at_index = np.zeros(len(list3d))
        print(tot_frames_for_no_norm)
        for inde, cand_point in enumerate(list3d):
            num_inside_image = inside_img_dict.get(tuple(cand_point))
            if num_inside_image:
                if float(num_inside_image)/tot_frames_for_no_norm >= thresh_coefs[0]:
                    keep_at_index[inde] = 1
    else:
        #check if thresh_coefs is of dtype float
        keep_at_index = np.zeros(len(list3d))

        for inde, cand_point in tqdm(enumerate(list3d)):
            num_inside_mask = inside_mask_dict.get(tuple(cand_point))
            num_inside_image = inside_img_dict.get(tuple(cand_point))
        
            if num_inside_image: #something is wrong if this is not true for all points...
                val =  num_inside_mask/num_inside_image #if normalize, then divide with num_inside_mask, else divide with 
            else:
                val = 0
            
            if val >= thresh_coefs:
                keep_at_index[inde] = 1
        
        return np.asarray(keep_at_index)

if reproj_masks == "XMEM_INT":
    path_to_masks = rf'/dtu/blackhole/07/155527/camproj/data/undistorted_images/XMEM_INT/Annotations/{object_name}/'
elif reproj_masks == "XMEM":
    path_to_masks = rf'/dtu/blackhole/07/155527/camproj/data/undistorted_images/XMEM/Annotations/{object_name}/'
else:
    print(f'masks for reprojection doesnt exist for {reproj_masks} \n EXITING')
    exit()


cam_json_path = rf'/dtu/blackhole/07/155527/camproj/data/GS_models/full_scene/{object_name}/{num_iterations}_round_{round_num}/cameras.json'

#inside_mask_dict, inside_img_dict = GS_reproj(cam_json_path, path_to_masks, scene)

mask_dict_path = f"/dtu/blackhole/07/155527/camproj/data/cache/{reproj_masks}/{object_name}/{num_iterations}/inside_mask_dict.pkl"
img_dict_path = f"/dtu/blackhole/07/155527/camproj/data/cache/{reproj_masks}/{object_name}/{num_iterations}/inside_img_dict.pkl"


# Ensure directories exist
os.makedirs(os.path.dirname(mask_dict_path), exist_ok=True)
os.makedirs(os.path.dirname(img_dict_path), exist_ok=True)

if os.path.exists(mask_dict_path) and os.path.exists(img_dict_path):

    # Load the dictionary from the file
    with open(mask_dict_path, "rb") as mask_file:
        inside_mask_dict = pickle.load(mask_file)
    with open(img_dict_path, "rb") as img_file:
        inside_img_dict = pickle.load(img_file)
else:
    inside_mask_dict, inside_img_dict = GS_reproj(cam_json_path, path_to_masks, scene)

    with open(mask_dict_path, "wb") as mask_file:
        pickle.dump(inside_mask_dict, mask_file)
    with open(img_dict_path, "wb") as img_file:
        pickle.dump(inside_img_dict, img_file)



hey = filter_fun_4_mesh(inside_mask_dict, inside_img_dict, scene.gaussians.get_xyz.cpu().detach().numpy(),thresh_coef)
#np.save(f"/dtu/blackhole/07/155527/camproj/{object_name}_mvg.npy", hey)
#hey = np.load(f"/dtu/blackhole/07/155527/camproj/{object_name}_mvg.npy")



list3d = scene.gaussians.get_xyz.cpu().detach().numpy()


mvg_result_list = [a for a in range(len(list3d)) if hey[a]]
#filtered_vertices = np.copy(list3d[vertices_keep_indx])

# Don't use .get... as it returns a property:scene.gaussians.get_xyz = scene.gaussians.get_xyz[vertices_keep_indx] doesnt work

# img = render(scene.getTestCameras()[0], scene.gaussians, pipeline.extract(args), background)["render"]

# #display 3xHxW image
# img = img.permute(1,2,0)
# img = img.detach().cpu().numpy()
# img = img - img.min()
# img = img / img.max()
# img = img * 255
# img = img.astype('uint8')
# import matplotlib.pyplot as plt
# # plt.imshow(img)
# # plt.show()
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# cv2.imwrite('bl_okay_im2_0_8.jpg',img)

#now render and save all test cameras/views assuming scene has been filtered
#import matplotlib.pyplot as plt

full_img_dir = fr"/dtu/blackhole/07/155527/camproj/data/undistorted_images/full_scene/JPEGImages/{object_name}/"
undistorted_mask_path = path_to_masks

print(f"num of gaussians before reprojection with segmentation masks is {scene.gaussians._xyz.shape[0]} and after is {len(mvg_result_list)}")

filter_name = f"{model_name}_MVG-{reproj_masks}_{thresh_coef}{white_str}"
scene = filter_gaussians_by_list_and_save(scene, pipeline, background,filter_name, full_img_dir,undistorted_mask_path, scene_info, mvg_result_list)

img = render(scene.getTestCameras()[0], scene.gaussians, pipeline.extract(args), background)["render"]

#display 3xHxW image
img = img.permute(1,2,0)
img = img.detach().cpu().numpy()
img = img - img.min()
img = img / img.max()
img = img * 255
img = img.astype('uint8')
import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.show()


# plt.hist(scene.gaussians._xyz[:,0].cpu().detach().numpy(), bins=100, color='r', alpha=0.5)
# plt.hist(scene.gaussians._xyz[:,1].cpu().detach().numpy(), bins=100, color='g', alpha=0.5)
# plt.hist(scene.gaussians._xyz[:,2].cpu().detach().numpy(), bins=100, color='b', alpha=0.5)
# plt.show()


S = torch.diag_embed(scene.gaussians._scaling)
quat = scene.gaussians._rotation

#normalize quaternion (paper says its important)
quat = quat / torch.norm(quat, dim=1).unsqueeze(1)

r = Rot.from_quat(quat.detach().cpu().numpy())
R = torch.tensor(r.as_matrix()).cuda().float()


sigmas_inv = torch.matmul(torch.matmul(torch.matmul(R,S),S.transpose(1,2)),R.transpose(1,2))

sigmas = torch.linalg.inv(sigmas_inv)


#print(f"cutoff of gaussian splat size is {percentile_value}, and mean splat size is {torch.mean(torch.abs(scene.gaussians._scaling))}")

# comparison = (torch.abs(scene.gaussians._scaling) < percentile_value).cpu().detach().numpy()
# # Check if any value in each row exceeds the threshold
# result = np.all(comparison, axis=1)

comparison = (torch.abs(sigmas) < rmv_sz).cpu().detach().numpy()
scale_result = ~np.any(~comparison, axis=(1,2))

# Convert the result to a list (if needed)
scale_result_list = scale_result.tolist()

import matplotlib.pyplot as plt

# plt.hist(scene.gaussians._xyz[:,0].cpu().detach().numpy(), bins=100, color='r', alpha=0.5)
# plt.hist(scene.gaussians._xyz[:,1].cpu().detach().numpy(), bins=100, color='g', alpha=0.5)
# plt.hist(scene.gaussians._xyz[:,2].cpu().detach().numpy(), bins=100, color='b', alpha=0.5)
# plt.show()


print(f"num of gaussians before removing big outliers (over {rmv_sz}) is {scene.gaussians._xyz.shape[0]} and after is {np.sum(scale_result_list)}")

filter_name = f"{model_name}_MVG-{reproj_masks}_{thresh_coef}_RMVXYZ_{rmv_sz}{white_str}"

scene = filter_gaussians_by_list_and_save(scene, pipeline, background,filter_name, full_img_dir,undistorted_mask_path, scene_info, scale_result_list,only_color=False)

mean = np.mean(scene.gaussians._xyz.cpu().detach().numpy(), axis=0)
sd = np.std(scene.gaussians._xyz.cpu().detach().numpy(), axis=0)
clean_result_list = (~np.any(np.abs(scene.gaussians._xyz.cpu().detach().numpy() - mean) > rmv_dist * sd, axis=1)).tolist()

print(f"num of gaussians before removing outliers {rmv_dist} standard deviations away from mean is {scene.gaussians._xyz.shape[0]} and after is {np.sum(clean_result_list)}")

filter_name =  f"{model_name}_MVG-{reproj_masks}_{thresh_coef}_RMVXYZ_{rmv_sz}_RMV{rmv_dist}STD{white_str}" 

scene = filter_gaussians_by_list_and_save(scene, pipeline, background,filter_name, full_img_dir,undistorted_mask_path, scene_info, clean_result_list,only_color=False,save=True)

# np.save(f"/dtu/blackhole/07/155527/camproj/tree_xyz_threshold_{percentile_value}_clean.npy", scene.gaussians._xyz.cpu().detach().numpy())


scene.gaussians.save_ply(f"/dtu/blackhole/07/155527/camproj/{object_name}_xyz_threshold_{rmv_sz}_clean.ply")


# img = render(scene.getTestCameras()[0], scene.gaussians, pipeline.extract(args), background)["render"]


#display 3xHxW image
# img = img.permute(1,2,0)
# img = img.detach().cpu().numpy()
# img = img - img.min()
# img = img / img.max()
# img = img * 255
# img = img.astype('uint8')
# import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.show()

# filter_name = f"{model_name}_MVG-{reproj_masks}_{thresh_coef}_RMVXYZ_{percentile_value}_RED"
# scene = filter_gaussians_by_list_and_save(scene, pipeline, background,filter_name, full_img_dir,undistorted_mask_path, scene_info, result_list,color_red=True)


# alpha_rmv = 0.1
# comparison = (torch.abs(scene.gaussians._opacity) > alpha_rmv).cpu().detach().numpy()
# result = np.all(comparison, axis=(1))
# # Convert the result to a list (if needed)
# result_list = result.tolist()

# filter_name = f"{model_name}_MVG-{reproj_masks}_{thresh_coef}_RMVXYZ_{percentile_value}_RMVALP_{alpha_rmv}"
# scene = filter_gaussians_by_list_and_save(scene, pipeline, background,filter_name, full_img_dir,undistorted_mask_path, scene_info, result_list)


## remove big gaussians, mvg - test renders of GS - 
# path_to_save = rf'/dtu/blackhole/07/155527/camproj/data/GS_models/MVG_{model_name}/test_{thresh_coef}_RMV_{percentile_value}_{object_name}/{round_num}/'
# print(f'saving to {path_to_save}')
# #if path to save does not exist, create it
# if not os.path.exists(path_to_save):
#     #create the directory
#     os.makedirs(path_to_save)
# render_all_test_cameras(scene, pipeline, background, path_to_save, full_img_dir,undistorted_mask_path, scene_info,False)



