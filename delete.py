# import math

# def calcular_otras_esquinas(x1, y1, x2, y2):
#     # Coordenadas de las dos esquinas opuestas
#     p1 = (x1, y1)
#     p2 = (x2, y2)
    
#     # Coordenadas del centro del cuadrado
#     cx = (x1 + x2) / 2
#     cy = (y1 + y2) / 2
    
#     # Vector desde el centro hasta una esquina
#     vx = x1 - cx
#     vy = y1 - cy
    
#     # Rotación de 90 grados
#     vx_rotado = -vy
#     vy_rotado = vx
    
#     # Coordenadas de las otras dos esquinas
#     x3 = cx + vx_rotado
#     y3 = cy + vy_rotado
#     x4 = cx - vx_rotado
#     y4 = cy - vy_rotado
    
#     return (x3, y3), (x4, y4)

# # Ejemplo de uso
# x1, y1 = 1366, 1384
# x2, y2 = 1349, 1810

# # esquinas = calcular_otras_esquinas(x1, y1, x2, y2)
# # print("Las coordenadas de las otras dos esquinas son:", esquinas)

from shapely.geometry import Polygon
import numpy as np

def calcular_cuadrado_con_centro(puntos, nuevo_centro):
    # Convertir puntos a un array numpy
    puntos = np.array(puntos)
    
    # Calcular el centro del cuadrado original
    centro_original = np.mean(puntos, axis=0)
    
    # Calcular el tamaño del cuadrado original (suponiendo que el cuadrado está orientado con los ejes)
    lado = np.linalg.norm(puntos[0] - puntos[1])
    
    # Calcular el ángulo de rotación del cuadrado (debería ser 0 si está alineado con los ejes)
    angulo = np.arctan2(puntos[1][1] - puntos[0][1], puntos[1][0] - puntos[0][0])
    
    # Crear una matriz de rotación para alinear el cuadrado con los ejes
    rotacion = np.array([
        [np.cos(angulo), -np.sin(angulo)],
        [np.sin(angulo), np.cos(angulo)]
    ])
    
    # Puntos relativos al centro original
    puntos_relativos = puntos - centro_original
    
    # Nuevos puntos relativos al nuevo centro
    nuevos_puntos_relativos = np.array([
        [-lado / 2, -lado / 2],
        [lado / 2, -lado / 2],
        [lado / 2, lado / 2],
        [-lado / 2, lado / 2]
    ])
    
    # Aplicar la rotación
    nuevos_puntos_relativos = nuevos_puntos_relativos @ rotacion.T
    
    # Trasladar al nuevo centro
    nuevos_puntos = nuevos_puntos_relativos + nuevo_centro
    
    return list(map(tuple, nuevos_puntos))

# Puntos del cuadrado original
puntos_originales = [(1245, 1111), (1532, 807), (1228, 520), (941, 824)]

# Nuevo centro deseado
nuevo_centro = (2846, 760)

# Calcular los nuevos puntos del cuadrado
nuevos_puntos = calcular_cuadrado_con_centro(puntos_originales, nuevo_centro)

print("Las coordenadas del nuevo cuadrado son:", nuevos_puntos)