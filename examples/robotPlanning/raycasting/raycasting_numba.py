# the ray casting algorithm is inspiret by:
# https://theshoemaker.de/2016/02/ray-casting-in-2d-grids/
# https://github.com/pfirsich/Ray-casting-test/blob/master/main.lua
# which is inspired by “A Fast Voxel Traversal Algorithm for Ray Tracing” by John Amanatides and Andrew Woo

import numba
from numba import jit, prange
from numba.typed import List
import torch
import numpy as np
pi = np.arccos(0) * 2  # 3.1415927410125732

@jit(nopython=True, fastmath = True) # Set "nopython" mode for best performance, equivalent to @njit
def getHelpers(cellSize, pos, rayDir):
    tile = int(np.floor(pos / cellSize)) + 1

    if rayDir == 0:
        dTile = 0
        dt = 0
        ddt = 0
    elif rayDir > 0:
        dTile = 1
        dt = ((tile + 0) * cellSize - pos) / rayDir
        ddt = dTile * cellSize / rayDir
    else:
        dTile = -1
        dt = ((tile - 1) * cellSize - pos) / rayDir
        ddt = dTile * cellSize / rayDir

    tile = tile - 1

    return tile, dTile, dt, ddt

@jit(nopython=True, fastmath = True) # Set "nopython" mode for best performance, equivalent to @njit
def castRayVectorDirALL(grid, rayStart, rayDir, maxdist):
    grid_shape = grid.shape
    grid_width = grid_shape[1]
    grid_height = grid_shape[0]

    #if flipped_y_axis:
    rayStartX = rayStart[0]
    rayStartY = grid_height - rayStart[1]
    rayDirX = rayDir[0]
    rayDirY = -rayDir[1]
    #else:
    # rayStartX = rayStart[0]
    # rayStartY = rayStart[1]
    # rayDirX = rayDir[0]
    # rayDirY = rayDir[1]

    cellSize = 1.
    tileX, dtileX, dtX, ddtX = getHelpers(cellSize, rayStartX, rayDirX)
    tileY, dtileY, dtY, ddtY = getHelpers(cellSize, rayStartY, rayDirY)
    t = 0.
    t_max = maxdist

    t_out = []  # List()
    tileY_out = []  # List()
    tileX_out = []  # List()
    gridValues = []  # List()
    if dtX == 0:
        while tileX >= 0 and tileX <= grid_width - 1 and tileY >= 0 and tileY <= grid_height - 1:
            if t > t_max:
                t = t_max
                if grid[tileY][tileX] > 0.:
                    t_out.append(t)
                    tileY_out.append(tileY)
                    tileX_out.append(tileX)
                    gridValues.append(grid[tileY][tileX])
                break

            if grid[tileY, tileX] > 0.:
                tileY_out.append(tileY)
                tileX_out.append(tileX)
                gridValues.append(grid[tileY][tileX])
                t_out.append(t)

            tileY = tileY + dtileY
            dt = dtY
            t = t + dt
            dtY = dtY + ddtY - dt
    elif dtY == 0:
        while tileX >= 0 and tileX <= grid_width - 1 and tileY >= 0 and tileY <= grid_height - 1:
            if t > t_max:
                t = t_max
                if grid[tileY][tileX] > 0.:
                    t_out.append(t)
                    tileY_out.append(tileY)
                    tileX_out.append(tileX)
                    gridValues.append(grid[tileY][tileX])
                break

            if grid[tileY][tileX] > 0.:
                tileY_out.append(tileY)
                tileX_out.append(tileX)
                gridValues.append(grid[tileY][tileX])
                t_out.append(t)

            tileX = tileX + dtileX
            dt = dtX
            t = t + dt
            dtX = dtX + ddtX - dt
    else:
        while tileX >= 0 and tileX <= grid_width - 1 and tileY >= 0 and tileY <= grid_height - 1:
            if t > t_max:
                t = t_max
                if grid[tileY][tileX] > 0.:
                    tileY_out.append(tileY)
                    tileX_out.append(tileX)
                    gridValues.append(grid[tileY][tileX])
                    t_out.append(t)
                break

            if grid[tileY][tileX] > 0.:
                tileY_out.append(tileY)
                tileX_out.append(tileX)
                gridValues.append(grid[tileY][tileX])
                t_out.append(t)

            if dtX < dtY:
                tileX = tileX + dtileX
                dt = dtX
                t = t + dt
                dtX = dtX + ddtX - dt
                dtY = dtY - dt
            else:
                tileY = tileY + dtileY
                dt = dtY
                t = t + dt
                dtX = dtX - dt
                dtY = dtY + ddtY - dt

    return t_out, tileY_out, tileX_out, gridValues

def castRayAngleDirALL(grid, rayStart, angle, maxdist):
    grid_ = grid.cpu().detach().numpy()
    rayStart_ = rayStart.cpu().detach().numpy()
    angle_ = angle.cpu().detach().numpy()
    maxdist_ = maxdist.cpu().detach().numpy()

    rayDir = np.array([np.cos(angle_), np.sin(angle_)])
    t, tileY, tileX, gridValues = castRayVectorDirALL(grid_, rayStart_, rayDir, maxdist_[0])

    t = torch.FloatTensor(t)
    tileY = torch.FloatTensor(tileY).long()
    tileX = torch.FloatTensor(tileX).long()
    gridValues = torch.FloatTensor(gridValues)

    return t, tileY, tileX, gridValues

@jit(nopython=True, fastmath = True, parallel=True) # Set "nopython" mode for best performance, equivalent to @njit
def _castMultipleRaysAngleDirALL(grid, rayStart, plate_context, N_beams, maxdist):
    I = len(plate_context)
    t = [[float(0),float(0)]] * I
    tileY = [[int(0),int(0)]] * I
    tileX = [[int(0),int(0)]] * I
    gridValues = [[float(0),float(0)]] * I
    angles = [float(0)] * I
    for i in prange(I):
        angles[i] = plate_context[i] * (pi * 2 / N_beams)
        rayDir = np.array([np.cos(angles[i]), np.sin(angles[i])])
        t[i], tileY[i], tileX[i], gridValues[i] = castRayVectorDirALL(grid, rayStart, rayDir, maxdist)
  
    return t, tileY, tileX, gridValues, angles

def castMultipleRaysAngleDirALL(grid, rayStart, plate_context, N_beams, maxdist):
    # simply conversion from torch to numpy and back...
    grid_ = grid.cpu().detach().numpy()
    rayStart_ = rayStart.cpu().detach().numpy()
    maxdist_ = maxdist.cpu().detach().numpy()

    typed_plate_context = List()
    [typed_plate_context.append(x) for x in plate_context]

    t_, tileY_, tileX_, gridValues_, angles_ = _castMultipleRaysAngleDirALL(grid_, rayStart_, typed_plate_context, N_beams, maxdist_[0])

    angles = torch.FloatTensor(angles_)
    t = []
    tileY = []
    tileX = []
    gridValues = []
    for i in range(len(angles)):
        t.append(torch.FloatTensor(t_[i]))
        tileY.append(torch.FloatTensor(tileY_[i]).long())
        tileX.append(torch.FloatTensor(tileX_[i]).long())
        gridValues.append(torch.FloatTensor(gridValues_[i]))
    return t, tileY, tileX, gridValues, angles



def lidar(map_grid, z_s, N_meas, maxdist):
    # return value of all non-zero cells together with the distance to that cell
    t = []
    gridValues = []
    tileX = []
    tileY = []
    angles = torch.arange(start=0, end=2*pi, step=pi*2/N_meas.item())
    for i in range(N_meas.item()):
        t_n, tileY_n, tileX_n, gridValues_n = castRayAngleDirALL(map_grid, z_s, angles[i], maxdist)
        t.append(t_n)
        gridValues.append(gridValues_n)
        tileX.append(tileX_n)
        tileY.append(tileY_n)
        
    tileX = torch.cat(tileX, 0)
    tileY = torch.cat(tileY, 0)
    return t, gridValues, angles, tileX, tileY