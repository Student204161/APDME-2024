import importlib
from MVG_utils import MVG_proj,filter_fun_4_mesh, load_meshes, get_filtered_mesh
import numpy as np, json, importlib, sys, itertools, os, cv2, copy
#os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# Fix pyglet 'shared environment' error
import matplotlib.pyplot as plt
from collections import Counter
#from preprocess_data.my_utils import Get3Dfrom2D_DepthMaps, forward_proj
from IPython.display import display, clear_output
from tqdm import tqdm
#import pyvista as pv
import pyrender
import trimesh
import open3d as o3d
import shutil
from datetime import datetime

method=sys.argv[1] #"GS
seg_objs=[sys.argv[2]]

# # Add the directory where `gaussian_renderer.py` is located to sys.path
# desired_directory = 'SuGaR'
# sys.path.insert(0, desired_directory)

# Now you can import the module
from SuGaR.gaussian_splatting.gaussian_renderer import render
import torchvision
from SuGaR.gaussian_splatting.utils.general_utils import safe_state
from argparse import ArgumentParser
from SuGaR.gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from SuGaR.gaussian_splatting.gaussian_renderer import GaussianModel

#!PATH=C:/Users/khali/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/Scripts:$PATH

thr_coefs=[[float(x) for x in sys.argv[3].split(',')]]  #[[0.95]]#[[0.93,0.5],[0.95,0.5]] # shoe round 5 to 7 missing for 0.9
dil_size=int(sys.argv[4]) #25
erode_size=int(sys.argv[5]) #20
roun=int(sys.argv[6])
working_dir=sys.argv[7]
os.chdir(working_dir)
#set this to > 0 if no normalizing w. respect to cameras that are looking. should be a list also I guess:
thr_no_norm_N = 0 # 0.75 * 375 (for plant), 0.75 * 
if method == "GS":
    print('GS used')
else:
    print('NERF used')
for thr_coef in thr_coefs: 
    for seg_obj in seg_objs: 
        print(os.getcwd()) 
        json_path = fr'{working_dir}/data/GS_models/XMEM/{seg_obj}/60000_round_1/cameras.json' if method =="GS" else f'{working_dir}/data/nerf_cameras/full_scene/{seg_obj}/transforms_train.json'
        #/dtu/blackhole/07/155527/camproj/data/GS_models/XMEM/plant/60000_round_1/output/refined_ply/plant/sugarfine_3Dgs60000_sdfestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.ply
        if method =="GS" :
            parser = ArgumentParser(description="Testing script parameters")
            model = ModelParams(parser, sentinel=True)
            pipeline = PipelineParams(parser)
            parser.add_argument("--iteration", default=-1, type=int)
            parser.add_argument("--skip_train", action="store_true")
            parser.add_argument("--skip_test", action="store_true")
            parser.add_argument("--quiet", action="store_true")

            # Add the following line to parse known arguments and ignore the rest
            args, unknown = parser.parse_known_args()
            args.model_path = rf"/dtu/blackhole/07/155527/camproj/data/GS_models/XMEM/{seg_obj}/60000_round_{roun}"
            args.iteration = 60000
            args.eval = True
            args.resolution=1
            args.source_path=rf'/dtu/blackhole/07/155527/camproj/data/colmap_info/{seg_obj}'
            args.quiet=False
            args.device='cuda'
            args.max_sh_degree=3
            print("Rendering " + args.model_path)
            # Initialize system state (RNG)
            def load_3DGS(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
                with torch.no_grad():
                    gaussians = GaussianModel(dataset.sh_degree)
                    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

                    return scene

            safe_state(args.quiet)
            scene = load_3DGS(model.extract(args), args.iteration, pipeline.extract(args))
            exit
        else:
            mesh_loc = fr"nerf_meshes_256/XMEM/{seg_obj}/0{roun}.ply"
            
        #mesh_loc = fr'data/GS_models/XMEM/{seg_obj}/60000_round_1/output/coarse_mesh/{seg_obj}/sugarmesh_3Dgs60000_sdfestim02_sdfnorm02_level03_decim1000000.ply' if method =="GS" else fr"nerf_meshes_256/XMEM/{seg_obj}/0{roun}.ply"
        undistorted_mask_path = fr'data/undistorted_images/XMEM/Annotations/{seg_obj}/'
        full_img_dir = fr"data/undistorted_images/full_scene/{seg_obj}/"
        test_masks_dir = fr"data/undistorted_images/test_images_uncorrected/masks/{seg_obj}" #fr"undistorted_corrected_GT_test_masks/XMEM/{seg_obj}" #doesn't matter here if its corrected or uncorrected though...
        test_jpgs_dir = fr"data/undistorted_images/test_images_uncorrected/masked_JPEGImages/{seg_obj}" #fr"undistorted_corrected_GT_test_images/XMEM/{seg_obj}/images"


        #edit these if using norm
        SEGTYPE = f"MVG_MESH_{method}_{thr_coef}_ero{erode_size}_dil{dil_size}_R{roun}"
        
        save_mask_path = f'data/{method}_models/MVG/{SEGTYPE}/Annotations/{seg_obj}/'
        save_jpgs_path = f'data/{method}_models/MVG/{SEGTYPE}/JPEGImages/{seg_obj}/'
        
        if not os.path.exists(save_mask_path):
            os.makedirs(save_mask_path)
        if not os.path.exists(save_jpgs_path):
            os.makedirs(save_jpgs_path)

        cur_time = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        cur_1 = datetime.now()

        with open(f'data/{method}_models/MVG/{SEGTYPE}/Annotations/{seg_obj}/time_log.log', 'a') as f:
            f.write(f"\nBeginning round {roun} of {SEGTYPE}, {seg_obj} at {cur_time}")

        train_json, mesh_o3d, mesh_pyrender, mesh_o3d_orig_coords = load_meshes(json_path=json_path,mesh_loc=mesh_loc,method=method)
        #o3d.visualization.draw_geometries([mesh_o3d])
        if len(thr_coef) == 1:
            mesh_pyrender_filtered, mesh_trimesh_filtered, mesh_o3d_filtered = get_filtered_mesh(mesh_o3d,json_path, undistorted_mask_path, method, thr_coef)
        else:
            mesh_pyrender_filtered_add, mesh_trimesh_filtered_add, mesh_o3d_filtered_add, mesh_pyrender_filtered_remove, mesh_trimesh_filtered_remove, mesh_o3d_filtered_remove = get_filtered_mesh(mesh_o3d,json_path, undistorted_mask_path, method, thr_coef)

        o3d.visualization.draw_geometries([mesh_o3d,mesh_o3d_filtered])
        c = 0  
        comp_list3d = []
        imgs_not_hit = []
        imgs_hit = []
        max_t_xyz = [0,0,0]


        dist_coefs = None # np.array([k1,k2,p1,p2,k3]); dist_coefs.shape = (5,1)

        rev_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

        results = []
        if method == "GS":
            tqdm_bar = tqdm(train_json)
            fov_y = 2*np.arctan(train_json[0]['height'] /(2*train_json[0][f'fy']))
            fov_x = 2*np.arctan(train_json[0]['width'] /(2*train_json[0][f'fx']))

            int_matrix = np.array([[train_json[0]['fx'], 0, train_json[0]['width']/2],
            [0, train_json[0]['fy'], train_json[0]['height']/2],
            [0, 0, 1]])
        else:
            tqdm_bar = tqdm(train_json['frames'])
            int_matrix = np.array([[train_json['fl_x'], 0, train_json['cx']],
                        [0, train_json['fl_y'], train_json['cy']],
                        [0, 0, 1]])
            
            fov_y = train_json['camera_angle_y']

        for im_data in tqdm_bar:
            c += 1

            if method == "GS":
                R_gaussian = np.asarray(im_data['rotation'])
                T_gaussian = np.asarray(im_data['position'])

                t_mat = np.hstack((R_gaussian, T_gaussian.reshape(-1, 1)))
                transf_matrix = np.vstack((t_mat,[0,0,0,1]))
                transf_matrix_inv = np.linalg.inv(transf_matrix)

                path_mask = f'{undistorted_mask_path}/' + im_data['img_name'] + '.png'
                filename = im_data['img_name'] + '.png'

            else:
                transf_matrix = np.array(im_data['transform_matrix'])
                path_mask = f'{undistorted_mask_path}/' + im_data['file_path'].split('/')[-1].replace('.jpg', '.png')
                filename = im_data['file_path'].split('/')[-1].replace('.jpg', '.png')
                
            
            #/work3/s204161/BachelorData/bachelor_data/masks/SAM/Annotations/white_knight/

            img_mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE) #
            full_img = cv2.imread( os.path.join(full_img_dir, filename.replace('.png', '.jpg')))

            #img_mask = cv2.undistort(img_mask,int_matrix,dist_coefs)

            h, w = img_mask.shape
            

            camera = pyrender.IntrinsicsCamera(fx=int_matrix[0,0], fy=int_matrix[1,1],cx=int_matrix[0,2], cy=int_matrix[1,2])
            #camera = pyrender.PerspectiveCamera(yfov=fov_y,aspectRatio=fov_x/fov_y)
            if len(thr_coef) == 1:
                scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1], bg_color=[0., 0., 0.])
                scene.add(mesh_pyrender_filtered)
                scene.add(camera, pose=transf_matrix)
            
                r = pyrender.OffscreenRenderer(w,h)

                _, density = r.render(scene)

                result = (density > 0).astype('int')
                result = (result*255).astype('uint8')
                dil_result = cv2.dilate(result, np.ones((dil_size, dil_size),np.uint8), iterations = 1)
                erode_result = cv2.erode(result, np.ones((erode_size, erode_size),np.uint8), iterations = 1)


                plo = np.zeros((h,w,3), dtype=np.uint8)                    
                #area that is in dil_result but not in erode_result = dil_result - erode_result
                unchanged_area = cv2.bitwise_and(dil_result, cv2.bitwise_not(erode_result))
                
                plo[:,:,0] = ~dil_result
                plo[:,:,1] = erode_result
                plo[:,:,2] = unchanged_area

                # plt.imshow(plo)
                # plt.show()
                
                result1 = cv2.bitwise_and(img_mask,dil_result)
                result = cv2.bitwise_or(result1, erode_result)

                # plt.imshow(img_mask)
                # plt.show()
                # plt.imshow(result)
                # plt.show()

                # result = (density > 0).astype('int')
                #result = (result*255).astype('uint8')
                result = (result.astype('float32'))/255
                for channel in range(3):
                    full_img[:,:,channel] = full_img[:,:,channel]*result

            else:
                scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1], bg_color=[0., 0., 0.])
                scene.add(mesh_pyrender_filtered_add)
                scene.add(camera, pose=transf_matrix)

                r = pyrender.OffscreenRenderer(w,h)

                _, density = r.render(scene)
                r.delete()

                result = (density > 0).astype('int')
                result = (result*255).astype('uint8')

                scene_rem = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1], bg_color=[0., 0., 0.])
                scene_rem.add(mesh_pyrender_filtered_remove)
                scene_rem.add(camera, pose=transf_matrix)

                r_rem = pyrender.OffscreenRenderer(w,h)

                _, density_rem = r_rem.render(scene_rem)
                r_rem.delete()

                result_rem = (density_rem > 0).astype('int')
                result_rem = (result_rem*255).astype('uint8')

                result1 = cv2.bitwise_and(img_mask, cv2.bitwise_not(result_rem))

                result2 = cv2.bitwise_or(result1, result)

                result = (result2.astype('float32'))/255
                
                for channel in range(3):
                    full_img[:,:,channel] = full_img[:,:,channel]*result

            result = (result*255).astype('uint8')
            #save mask
            file_mask_path = os.path.join(save_mask_path,filename)
            cv2.imwrite(file_mask_path,result)

            file_jpgs_path = os.path.join(save_jpgs_path,filename.replace('.png', '.jpg'))

            #save masked image
            cv2.imwrite(file_jpgs_path,full_img)

            tqdm_bar.set_description(f"... Round {roun}, writing MVG-corrected masks")
        
        #copy test images and copy test masks (Only important for Gaussian Splatting pipeline though)
        test_masks = os.listdir(test_masks_dir)
        test_jpgs = os.listdir(test_jpgs_dir)
        for test_mask in test_masks:
            test_mask_path = os.path.join(test_masks_dir,test_mask)
            save_test_mask_path = os.path.join(save_mask_path, test_mask)
            shutil.copy(test_mask_path,save_test_mask_path)
        for test_jpg in test_jpgs:
            test_jpg_path = os.path.join(test_jpgs_dir,test_jpg)
            save_test_jpg_path = os.path.join(save_jpgs_path, test_jpg)
            shutil.copy(test_jpg_path,save_test_jpg_path)
        
        
        fin_time = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

        cur_2 = datetime.now()
        dif_time = cur_2 - cur_1

        print(f"\n Finished round {roun} of {SEGTYPE}, {seg_obj} at {cur_time} at {fin_time}, diff time is {dif_time}")
        with open(f'data/{method}_models/MVG/{SEGTYPE}/Annotations/{seg_obj}/time_log.log', 'a') as f:
            f.write(f"\n Finished round {roun} of {SEGTYPE}, {seg_obj} at {cur_time} at {fin_time}, diff time is {dif_time}")

