import bpy
import pathlib

import scipy.io as sio

TEMP_DATA = {}

def get_cache_matrix(matrix_path: pathlib.Path)->dict:
    cache: dict = TEMP_DATA.get(matrix_path.stem, None)
    if(not cache):
        if(matrix_path.exists()):
            mat_file: dict = sio.loadmat(f'{matrix_path}')
            TEMP_DATA[matrix_path.stem] = mat_file
            cache = mat_file
            return cache
        else:
            return {}
    return cache

def is_matrix_loaded(matrix_name: str)->bool:
    if(TEMP_DATA.get(matrix_name)):
        return True
    return False

def load_matrix_to_cache(matrix_path: pathlib.Path)->dict:
    return get_cache_matrix(matrix_path)

def get_cache_matrix_name(matrix_name: str)->dict:
    return TEMP_DATA.get(matrix_name)