import numpy as np, json, importlib, sys, itertools, os, cv2,copy
import matplotlib.pyplot as plt
from collections import Counter
#from preprocess_data.my_utils import Get3Dfrom2D_DepthMaps, forward_proj
#from IPython.display import display, clear_output
from tqdm import tqdm
import open3d as o3d
import pyrender
import trimesh
import pymeshfix



def MVG_proj(list3d, json_path, path_to_masks,synth_method):

    if synth_method == "NERF":
        with open(json_path) as f:
            cam_json = json.load(f)
        
        print(json_path)

        #sort frames by order, 0000 first...
        cam_json['frames'] = sorted(cam_json['frames'], key=lambda x: int(x["file_path"].split("/")[-1].split(".")[0]))

        C_x = cam_json['cx']  # int(img1.shape[0]/2)
        C_y = cam_json['cy']  # int(img1.shape[1]/2)
        f_x = cam_json['fl_x']
        f_y = cam_json['fl_y']


        int_matrix = np.array([[-f_x, 0, C_x],
                        [0, f_y, C_y],
                        [0, 0, 1]])

        c = 0
        dist_coefs = None #np.array([k1,k2,p1,p2,k3]); dist_coefs.shape = (5,1)
        tqdm_bar = tqdm(cam_json['frames'])
        inside_mask_dict = {tuple(point): 0 for point in list3d}
        inside_image_list = np.zeros(len(list3d))
        counter = Counter(inside_mask_dict)


        for im_data in tqdm_bar:
            transf_matrix = np.array(im_data['transform_matrix'])

            transf_matrix_inv = np.linalg.inv(transf_matrix)

            R = transf_matrix_inv[:3,:3]
            t = transf_matrix_inv[:3,3]

            rvec, _ = cv2.Rodrigues(R)
            cam_pos = t

            path_mask = path_to_masks +  im_data['file_path'].split('/')[-1].replace('.jpg', '.png') 
            img_mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE) # 
            #img_mask = cv2.undistort(img_mask,int_matrix,dist_coefs)
            #dilate mask just a little bit in case the round off removes vertices where it shouldn't
            # kernel = np.ones((1, 1), np.uint8)
            # img_mask = cv2.dilate(img_mask, kernel, iterations=1)
            
            h, w = img_mask.shape
            fast_uv = cv2.projectPoints(np.asarray(list3d), rvec, t, int_matrix, dist_coefs)

            fast_uv_rounded = np.asarray(np.round(fast_uv[0].reshape(len(fast_uv[0]),2)),dtype='int')

            inside_img = np.all([fast_uv_rounded[:,0] >= 0,fast_uv_rounded[:,0] < w,0 <= fast_uv_rounded[:,1],fast_uv_rounded[:,1] < h], axis=0)
            #fast_uv_rounded_copy = np.copy(fast_uv_rounded)
            fast_uv_rounded[~inside_img] = 0
            inside_image_list += inside_img


            inside_mask = np.full(len(fast_uv_rounded),False)
            inside_mask = img_mask[fast_uv_rounded[:,1],fast_uv_rounded[:,0]] > 0
            inside_mask[~inside_img] = False
            inside_mask_xyz = list3d[inside_mask]

            inside_mask_xyz = tuple(map(tuple, inside_mask_xyz))
            counter.update(inside_mask_xyz)
            # Visualize the mask and the cleaned up points
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
        
    elif synth_method == "GS": 
        with open(json_path) as f:
            cam_json = json.load(f)

        test_imgs_num = 0
        for i in range(len(cam_json)):
            if i % 8 == 0:
                test_imgs_num +=1

        cam_json = cam_json[test_imgs_num:]

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

    return None


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
        if isinstance(thresh_coefs, float):
            keep_at_index = np.zeros(len(list3d))

            for inde, cand_point in enumerate(list3d):
                num_inside_mask = inside_mask_dict.get(tuple(cand_point))
                num_inside_image = inside_img_dict.get(tuple(cand_point))
            
                if num_inside_image: #something is wrong if this is not true for all points...
                    val =  num_inside_mask/num_inside_image #if normalize, then divide with num_inside_mask, else divide with 
                else:
                    val = 0
                
                if val >= thresh_coefs:
                    keep_at_index[inde] = 1
            
            return np.asarray(keep_at_index)
        elif len(thresh_coefs) == 2:
            add_at_index = np.zeros(len(list3d)) #upper threshold
            remove_at_index = np.zeros(len(list3d)) #lower threshold

            for inde, cand_point in enumerate(list3d):
                num_inside_mask = inside_mask_dict.get(tuple(cand_point))
                num_inside_image = inside_img_dict.get(tuple(cand_point))
                if num_inside_image: #something is wrong if this is not true for all points...
                    val =  num_inside_mask/num_inside_image #if normalize, then divide with num_inside_mask, else divide with 
                else:
                    val= 0
        
                if val >= thresh_coefs[0]: # if 90% - if a 2d point in the original mask is 0/1, but in MVG corrected mask it's negative, then 
                    add_at_index[inde] = 1
                elif val <= thresh_coefs[1]: #e.g. 10% - we will create a mesh with only the points that are in the mask less 10% of the time
                    remove_at_index[inde] = 1
            
            return np.asarray(add_at_index), np.asarray(remove_at_index)
        else:
            #thresh must be a list of 1 or 2 elements
            print("thresholds must be a list of 1 or 2 elements")
            return None
    
def filter_mesh(all_vertices,all_vertex_colors,all_triangles, vertices_keep_ind):
    vertices_keep_indx = [a for a in range(len(all_vertices)) if vertices_keep_ind[a]]
    filtered_vertices = np.copy(all_vertices[vertices_keep_indx])
    filtered_colors =  np.copy(all_vertex_colors[vertices_keep_indx])
        

    ## edit triangles/faces of the mesh
    vertex_num = 0
    vertex_num_dict = {}
    new_triangles = []
    for index, vertex in enumerate(all_vertices):
        if vertices_keep_ind[index]:
            vertex_num_dict[index] = vertex_num
            vertex_num += 1

    for triangle in all_triangles:
        new_triangle = [] 
        for triangle_corner in triangle:
            if vertices_keep_ind[triangle_corner]:
                new_triangle.append(vertex_num_dict[triangle_corner])

        if len(new_triangle) == 3:
            new_triangles.append(new_triangle)

    filtered_triangles = np.asarray(new_triangles)

    mesh_o3d_filtered = o3d.geometry.TriangleMesh()
    mesh_o3d_filtered.vertices = o3d.utility.Vector3dVector(filtered_vertices)
    mesh_o3d_filtered.vertex_colors = o3d.utility.Vector3dVector(filtered_colors)
    mesh_o3d_filtered.triangles = o3d.utility.Vector3iVector(filtered_triangles)
    return mesh_o3d_filtered



def nerf_pos_to_gl(vertices):
        #also works for cv2 proj., though f_x has to be taken negative sign of.
        vertices_copy = np.copy(vertices)
        # Flip the X, Y, and Z axes (change the order to suit your needs)
        #vertices[:, 0] = vertices[:, 0]
        #vertices[:, 1] = -vertices[:, 2]
        
        vertices[:, 0] = vertices_copy[:, 1]
        vertices[:, 1] = vertices_copy[:, 2]
        vertices[:, 2] = vertices_copy[:, 0]
        
        vertices_copy = np.copy(vertices)

        vertices[:, 0] = vertices_copy[:, 2]
        vertices[:, 2] = vertices_copy[:, 0]

        vertices_copy = np.copy(vertices)

        vertices[:, 0] = vertices_copy[:, 1]
        vertices[:, 1] = vertices_copy[:, 0]

        return vertices

def flip_filtered_nerf_mesh(vertices):
        #also works for cv2 proj., though f_x has to be taken negative sign of.
        vertices_copy = np.copy(vertices)
        # Flip the X, Y, and Z axes (change the order to suit your needs)
        #vertices[:, 0] = vertices[:, 0]
        #vertices[:, 1] = -vertices[:, 2]
        
        vertices[:, 0] = vertices_copy[:, 2]
        vertices[:, 1] = vertices_copy[:, 0]
        vertices[:, 2] = vertices_copy[:, 1]
        
        vertices_copy = np.copy(vertices)

        vertices[:, 0] = vertices_copy[:, 2]
        vertices[:, 2] = vertices_copy[:, 0]

        vertices_copy = np.copy(vertices)

        vertices[:, 2] = vertices_copy[:, 1]
        vertices[:, 1] = vertices_copy[:, 2]

        return vertices


def load_meshes(json_path, mesh_loc, method):

    with open(json_path) as f:
        train_json = json.load(f)

    if method == "GS": # if Gaussian Splatting
        test_imgs_num = 0

        for i in range(len(train_json)):
            if i % 8 == 0:
                test_imgs_num +=1

        train_json = train_json[test_imgs_num:]

        #point_cloud = o3d.io.read_point_cloud(mesh_loc)
        mesh_o3d = o3d.io.read_triangle_mesh(mesh_loc)
        mesh_o3d_orig_coords = copy.deepcopy(mesh_o3d) #o3d.io.read_triangle_mesh(mesh_loc)

        mesh_trimesh = trimesh.Trimesh(vertices=mesh_o3d.vertices, faces=mesh_o3d.triangles)
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=False)

        # point_cloud_copy = o3d.io.read_point_cloud(mesh_loc)
        # np.asarray(point_cloud.points)[:,1] *= -1
        # np.asarray(point_cloud.points)[:,2] *= -1
        # np.asarray(point_cloud.points).T
        
        # mesh_o3d , uh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=8)

        # vertices = np.asarray(mesh_o3d.vertices)
        # faces = np.asarray(mesh_o3d.triangles)
        # homogeneous_vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
        # transformation_matrix = np.array([
        # [1, 0, 0, 0],
        # [0, -1, 0, 0],
        # [0, 0, -1, 0]
        # ])
        # # Transform the mesh vertices
        #transformed_vertices = np.dot(transformation_matrix, vertices.T).T
        
        
        #mesh_trimesh = trimesh.Trimesh(vertices=transformed_vertices[:, :3], faces=faces)
        # mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
    elif method == "NERF":
        mesh_o3d = o3d.io.read_triangle_mesh(mesh_loc)
        mesh_o3d_orig_coords = copy.deepcopy(mesh_o3d) #o3d.io.read_triangle_mesh(mesh_loc)
        # Access the vertices
        vertices = np.asarray(mesh_o3d.vertices)
        # vertices_copy = np.copy(vertices)
        # Flip the X, Y, and Z axes (change the order to suit your needs)
        #vertices[:, 0] = vertices[:, 0]
        #vertices[:, 1] = -vertices[:, 2]
        
        # vertices[:, 0] = vertices_copy[:, 1]
        # vertices[:, 1] = vertices_copy[:, 2]
        # vertices[:, 2] = vertices_copy[:, 0]
        
        # vertices_copy = np.copy(vertices)

        # vertices[:, 0] = vertices_copy[:, 2]
        # vertices[:, 2] = vertices_copy[:, 0]

        # vertices_copy = np.copy(vertices)

        # vertices[:, 0] = vertices_copy[:, 1]
        # vertices[:, 1] = vertices_copy[:, 0]
            
        vertices = nerf_pos_to_gl(vertices)

        # Update the vertices in the Trimesh
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        
        #filtered_mesh = trimesh.Trimesh(vertices=np.asarray(mesh_o3d.vertices)[vertices_keep_indx], faces=filtered_triangles)

        mesh_trimesh = trimesh.Trimesh(vertices=mesh_o3d.vertices, faces=mesh_o3d.triangles)

        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=False)

    return train_json, mesh_o3d, mesh_pyrender, mesh_o3d_orig_coords


def get_filtered_mesh(mesh_o3d,json_path, undistorted_mask_path, method, coef,no_norm_tot_N=0):
    all_vertices = copy.deepcopy(np.asarray(mesh_o3d.vertices))
    all_vertex_colors = copy.deepcopy(np.asarray(mesh_o3d.vertex_colors))
    all_triangles = copy.deepcopy(np.asarray(mesh_o3d.triangles))

    
    inside_mask_dict, inside_img_dict = MVG_proj(all_vertices, json_path, undistorted_mask_path, method)

    #save inside_mask_dict and inside_img_dict
    #np.save('inside_mask_dict.npy', inside_mask_dict)
    #np.save('inside_img_dict.npy', inside_img_dict)
    #load inside_mask_dict and inside_img_dict
    # inside_mask_dict = np.load('inside_mask_dict.npy', allow_pickle=True).item()
    # inside_img_dict = np.load('inside_img_dict.npy', allow_pickle=True).item()


    if len(coef) == 1:
        vertices_keep_ind = filter_fun_4_mesh(inside_mask_dict, inside_img_dict, all_vertices, coef[0], no_norm_tot_N)

        mesh_o3d_filtered = filter_mesh(all_vertices,all_vertex_colors,all_triangles, vertices_keep_ind)

        mesh_o3d_filtered = mesh_fill_small_boundaries(mesh_o3d_filtered)


        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh_o3d_filtered.cluster_connected_triangles())
        

        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)

        triangles_to_remove = cluster_n_triangles[triangle_clusters] < 1000
        mesh_o3d_filtered.remove_triangles_by_mask(triangles_to_remove)


        mesh_trimesh_filtered = trimesh.Trimesh(vertices=mesh_o3d_filtered.vertices, faces=mesh_o3d_filtered.triangles, vertex_colors=mesh_o3d_filtered.vertex_colors)

        mesh_pyrender_filtered = pyrender.Mesh.from_trimesh(mesh_trimesh_filtered, smooth=False)

        return mesh_pyrender_filtered, mesh_trimesh_filtered, mesh_o3d_filtered
    else:
        vertices_add_ind, vertices_remove_ind = filter_fun_4_mesh(inside_mask_dict, inside_img_dict, all_vertices, coef, no_norm_tot_N)
        vertices_add_indx = [a for a in range(len(all_vertices)) if vertices_add_ind[a]]
        vertices_remove_indx = [a for a in range(len(all_vertices)) if vertices_remove_ind[a]]

        filtered_vertices_add, filtered_colors_add = np.copy(all_vertices[vertices_add_indx]), np.copy(all_vertex_colors[vertices_add_indx])
        filtered_vertices_remove, filtered_colors_remove = np.copy(all_vertices[vertices_remove_indx]), np.copy(all_vertex_colors[vertices_remove_indx])
        

        ## edit triangles/faces of the mesh

        mesh_o3d_filtered_add = filter_mesh(all_vertices,all_vertex_colors,all_triangles, vertices_add_ind)
        mesh_o3d_filtered_remove = filter_mesh(all_vertices,all_vertex_colors,all_triangles, vertices_remove_ind)
        

        mesh_o3d_filtered_add = mesh_fill_small_boundaries(mesh_o3d_filtered_add)


        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh_o3d_filtered_add.cluster_connected_triangles())
        

        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)

        triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
        mesh_o3d_filtered_add.remove_triangles_by_mask(triangles_to_remove)


        mesh_trimesh_filtered_add = trimesh.Trimesh(vertices=mesh_o3d_filtered_add.vertices, faces=mesh_o3d_filtered_add.triangles, vertex_colors=mesh_o3d_filtered_add.vertex_colors)
        mesh_trimesh_filtered_remove = trimesh.Trimesh(vertices=mesh_o3d_filtered_remove.vertices, faces=mesh_o3d_filtered_remove.triangles, vertex_colors=mesh_o3d_filtered_remove.vertex_colors)
        
        mesh_pyrender_filtered_add = pyrender.Mesh.from_trimesh(mesh_trimesh_filtered_add, smooth=False)
        mesh_pyrender_filtered_remove = pyrender.Mesh.from_trimesh(mesh_trimesh_filtered_remove, smooth=False)

        return mesh_pyrender_filtered_add, mesh_trimesh_filtered_add, mesh_o3d_filtered_add, mesh_pyrender_filtered_remove, mesh_trimesh_filtered_remove, mesh_o3d_filtered_remove

def get_filtered_mesh_BO(mesh_o3d,json_path, undistorted_mask_path, method, sweep_config, train_json, full_img_dir):
    all_vertices = copy.deepcopy(np.asarray(mesh_o3d.vertices))
    all_vertex_colors = copy.deepcopy(np.asarray(mesh_o3d.vertex_colors))
    all_triangles = copy.deepcopy(np.asarray(mesh_o3d.triangles))

    import wandb

    inside_mask_dict, inside_img_dict = MVG_proj(all_vertices, json_path, undistorted_mask_path, method)


def first_frame_loss(inside_mask_dict, inside_img_dict, all_vertices, all_vertex_colors, all_triangles, no_norm_tot_N, 
                     sweep_config, train_json, full_img_dir):

    with wandb.init(config=sweep_config):
        config = wandb.config
        SEGTYPE = f"MVG_MESH_{config.name}_{config.coef}_norm_v5_bayes_R{config.run_id}"
        save_mask_path = f'masks/{SEGTYPE}/Annotations/{config.seg_obj}/'
        save_jpgs_path = f'masks/{SEGTYPE}/JPEGImages/{config.seg_obj}/'
        full_img_dir = fr"undistorted_full_images/{config.seg_obj}"
        if not os.path.exists(save_mask_path):
            os.makedirs(save_mask_path)
        if not os.path.exists(save_jpgs_path):
            os.makedirs(save_jpgs_path)

        if len(config.coef) == 1:
            vertices_keep_ind = filter_fun_4_mesh(inside_mask_dict, inside_img_dict, all_vertices, config.coef, no_norm_tot_N)
            mesh_o3d_filtered = filter_mesh(all_vertices,all_vertex_colors,all_triangles, vertices_keep_ind)
            mesh_o3d_filtered = mesh_fill_small_boundaries(mesh_o3d_filtered)

            with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
                triangle_clusters, cluster_n_triangles, cluster_area = (
                mesh_o3d_filtered.cluster_connected_triangles())
            

            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            cluster_area = np.asarray(cluster_area)

            triangles_to_remove = cluster_n_triangles[triangle_clusters] < 1000
            mesh_o3d_filtered.remove_triangles_by_mask(triangles_to_remove)


            mesh_trimesh_filtered = trimesh.Trimesh(vertices=mesh_o3d_filtered.vertices, faces=mesh_o3d_filtered.triangles, vertex_colors=mesh_o3d_filtered.vertex_colors)
            mesh_pyrender_filtered = pyrender.Mesh.from_trimesh(mesh_trimesh_filtered, smooth=False)
        else:
            vertices_add_ind, vertices_remove_ind = filter_fun_4_mesh(inside_mask_dict, inside_img_dict, all_vertices, config.coef, no_norm_tot_N)

            mesh_o3d_filtered_add = filter_mesh(all_vertices,all_vertex_colors,all_triangles, vertices_add_ind)
            mesh_o3d_filtered_remove = filter_mesh(all_vertices,all_vertex_colors,all_triangles, vertices_remove_ind)
            

            mesh_o3d_filtered_add = mesh_fill_small_boundaries(mesh_o3d_filtered_add)

            with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
                triangle_clusters, cluster_n_triangles, cluster_area = (
                mesh_o3d_filtered_add.cluster_connected_triangles())
            

            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            cluster_area = np.asarray(cluster_area)

            triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
            mesh_o3d_filtered_add.remove_triangles_by_mask(triangles_to_remove)


            mesh_trimesh_filtered_add = trimesh.Trimesh(vertices=mesh_o3d_filtered_add.vertices, faces=mesh_o3d_filtered_add.triangles, vertex_colors=mesh_o3d_filtered_add.vertex_colors)
            mesh_trimesh_filtered_remove = trimesh.Trimesh(vertices=mesh_o3d_filtered_remove.vertices, faces=mesh_o3d_filtered_remove.triangles, vertex_colors=mesh_o3d_filtered_remove.vertex_colors)
            
            mesh_pyrender_filtered_add = pyrender.Mesh.from_trimesh(mesh_trimesh_filtered_add, smooth=False)
            mesh_pyrender_filtered_remove = pyrender.Mesh.from_trimesh(mesh_trimesh_filtered_remove, smooth=False)


        int_matrix = np.array([[train_json['fl_x'], 0, train_json['cx']],
                    [0, train_json['fl_y'], train_json['cy']],
                    [0, 0, 1]])
        
        im_data = train_json['frames'][0]

        transf_matrix = np.array(im_data['transform_matrix'])
        path_mask = f'C:/Users/khali/Desktop/share_mesh/undistorted_masks/{config.seg_obj}/' + im_data['file_path'].split('/')[-1].replace('.jpg', '.png')
        filename = im_data['file_path'].split('/')[-1].replace('.jpg', '.png')
            
        img_mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE) #
        full_img = cv2.imread(os.path.join(full_img_dir, filename.replace('.png', '.jpg')))

        h, w = img_mask.shape
        

        camera = pyrender.IntrinsicsCamera(fx=int_matrix[0,0], fy=int_matrix[1,1],cx=int_matrix[0,2], cy=int_matrix[1,2])
        #camera = pyrender.PerspectiveCamera(yfov=fov_y,aspectRatio=fov_x/fov_y)
        if len(config.coef) == 1:
            scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1], bg_color=[0., 0., 0.])
            scene.add(mesh_pyrender_filtered)
            scene.add(camera, pose=transf_matrix)
        
            r = pyrender.OffscreenRenderer(w,h)

            _, density = r.render(scene)

            result = (density > 0).astype('int')

            result = (result*255).astype('uint8')

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

        #calculate loss (PSNR)
        


        

    

    


def mesh_fill_small_boundaries(mesh):
    # fill small holes
    tin = pymeshfix.PyTMesh()    
    tin.load_array(np.asarray(mesh.vertices), np.asarray(mesh.triangles)) 
    tin.fill_small_boundaries(0.005)
    new_vertices, new_triangles = tin.return_arrays()
    mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    return mesh
