import numpy as np


def coated_conductor(
    conductor_roughness, 
    interface_roughness, 
    albedo, 
    reflectance, 
    thickness,
):
    textures = []
    material = "coatedconductor"
    params = [
        {
            'name': "float conductor.roughness",
            'value': np.array([ conductor_roughness ], dtype=np.float32),
        },
        {
        
            'name': "float interface.roughness",
            'value': np.array([ interface_roughness ], dtype=np.float32),
        },
        {
        
            'name': "rgb albedo",
            'value': np.array(albedo, dtype=np.float32),
        },
        {
        
            'name': "rgb reflectance",
            'value': np.array(reflectance, dtype=np.float32),
        },
        {
        
            'name': "float thickness",
            'value': np.array([ thickness ], dtype=np.float32),
        },
    ]
    return textures, material, params


def coated_conductor_texture(albedo_file, rough_file):
    textures = [
        f"Texture \"albedo\" \"spectrum\" \"imagemap\" \"string filename\" [ \"{albedo_file.name}\" ]",
        f"Texture \"rough\" \"float\" \"imagemap\" \"string filename\" [ \"{rough_file.name}\" ]"
    ]
    material = "coatedconductor"
    params = [
        {
            'name': "texture albedo",
            'value': "[ \"albedo\" ]",
        },
        {
        
            'name': "texture reflectance",
            'value': "[ \"albedo\" ]",
        },
        {
        
            'name': "texture conductor.roughness",
            'value': "[ \"rough\" ]",
        },
        {
        
            'name': "texture interface.roughness",
            'value': "[ \"rough\" ]",
        },
    ]
    return textures, material, params


def coated_diffuse(roughness, albedo, reflectance, thickness):
    textures = []
    material = "coateddiffuse"
    params = [
        {
        
            'name': "float roughness",
            'value': np.array([ roughness ], dtype=np.float32),
        },
        {
        
            'name': "rgb albedo",
            'value': np.array(albedo, dtype=np.float32),
        },
        {
        
            'name': "rgb reflectance",
            'value': np.array(reflectance, dtype=np.float32),
        },
        {
        
            'name': "float thickness",
            'value': np.array([ thickness ], dtype=np.float32),
        },
    ]
    return textures, material, params


def diffuse(reflectance=np.array([1.0, 1.0, 1.0])):
    textures = []
    material = "diffuse"
    params = [
        {
            'name': "rgb reflectance",
            'value': np.array(reflectance, dtype=np.float32),
        }
    ]
    return textures, material, params