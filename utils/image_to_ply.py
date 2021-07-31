import os
from glob import glob
from posix import listdir
import open3d as o3d
import pydicom
import numpy as np
def process_images(path:str):
    for modes in ["test","train"]:
        for indiv in os.listdir(os.join(path,modes)):
            # process each of the 4 data sources
            img_types = ["FLAIR","T1w","T1wCE","T2w"]
            for type in img_types:
                file_path = os.join(path,modes,indiv,type)
                depth = 0
                for files in os.listdir(file_path):
                    dicom = pydicom.read_file(os.join(file_path,files))
                    data = dicom.pixel_array
                    if np.min(data)==np.max(data):
                        pass
                    else: 
                        # normalize through 0,1 to prevent gradient overflow
                        data = data - np.min(data)
                        if np.max(data) != 0:
                            data = data / np.max(data)
                    
                    # now need to process data to x,y,z,intensity
                    depth+=1
        # the depth will be the image nb
        # x,y are the pixel
        # no RGB, but intensity 
        pass
        # Pass numpy array to Open3D.o3d.geometry.PointCloud and visualize
        xyz = np.random.rand(100, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        o3d.io.write_point_cloud("../data/data.ply", pcd)
        o3d.visualization.draw_geometries([pcd])
        return 

    

if __name__ == '__main__':
    path = "../data"
    process_images(path)