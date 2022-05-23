# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 23:38:19 2018

@author: Yury
"""
import os
import zipfile


def path_that_not_exist(path, create = False):
    r"""Checks if existing path doesn't exist - and if exist 
    add new path (so old path isn't rewritten). It works for folders and files
    -------
    Args:
    - absolute path to file / folder
    - create - whether or note to create folder (not applicable to files)
    Note: if folder it doesn't matter whether there is "/" or not at the end of path
    -------
    Returns:
    - returns new path
    """
    #-1 folder without "/", 0 - folder with "/", 1 - file extension
    path_file = -1
    file_extension = ''
    
    if path.endswith('/'): 
        path_file = 0
        path= path[:-1]
    elif '.' in path: 
        path_file = 1
        [path, file_extension] = path.split('.')
        file_extension = '.' + file_extension

    new_path = path

    for i in np.arange(1000):
        if os.path.exists(new_path + file_extension):
            new_path = path + '_' + str(i)
        else: break

    if path_file==0:
        new_path+= '/'
    elif path_file == 1:
        new_path += file_extension
    
    if create and (path_file==0 or path_file==-1):
        os.makedirs(new_path)

    return new_path

import datetime
td=datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")


def pycCleanup(directory,path):
    for filename in directory: 
        if     filename[-3:] == 'pyc': 
            print ('- ' + filename )
            os.remove(path+os.sep+filename) 
        elif os.path.isdir(path+os.sep+filename): 
            pycCleanup(os.listdir(path+os.sep+filename),path+os.sep+filename)

directory = os.listdir('.')
print('Deleting pyc files recursively in: '+ str(directory)) 
pycCleanup(directory,'.')
        
def save_src_to_zip(save_path, exclude_folders = [], dname="src", td=td):
#    goal_dir = ""
#    save_path=os.path.dirname(os.getcwd())+'\\src_zip\\'
    
    
    zf = zipfile.ZipFile(save_path+"src_"+ td+".zip", "w")
#    print(goal_dir + 'src/')
#    for dirname, subdirs, files in os.walk(goal_dir + 'src/'):
    exclude_folders = []
    for dirname, subdirs, files in os.walk("..\\" +dname+"\\"):
        for ef in exclude_folders:
            if (ef in subdirs):
                subdirs.remove(ef)
                #print (ef)
        zf.write(os.path.dirname(os.getcwd()),dname)
       
        #print (dirname)
        for filename in files:
            file = os.path.join(os.path.dirname(os.getcwd())+"\\"+dname+"\\", filename)
            #from to
            zf.write(file,dname +"\\"+filename)
            print(dirname, filename)
    zf.close()