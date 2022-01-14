# the ray casting algorithm is inspiret by:
# https://theshoemaker.de/2016/02/ray-casting-in-2d-grids/
# https://github.com/pfirsich/Ray-casting-test/blob/master/main.lua
# which is inspired by “A Fast Voxel Traversal Algorithm for Ray Tracing” by John Amanatides and Andrew Woo

import torch
import pyro
from pyro.contrib.autoname import scope
import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistributionMixin

from misc.torch_truncnorm.TruncatedNormal_Modified import TruncatedNormal

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # 3.1415927410125732


@torch.jit.script
def getHelpers(cellSize, pos, rayDir):
    tile = torch.floor(pos / cellSize).long() + 1

    if rayDir == 0:
        dTile = torch.tensor([0], dtype=torch.long)
        dt = torch.tensor([0])
        ddt = torch.tensor([0])
    elif rayDir > 0:
        dTile = torch.tensor([1], dtype=torch.long)
        dt = ((tile + torch.tensor([0.])) * cellSize - pos) / rayDir
        ddt = dTile * cellSize / rayDir
    else:
        dTile = torch.tensor([-1], dtype=torch.long)
        dt = ((tile - torch.tensor([1.])) * cellSize - pos) / rayDir
        ddt = dTile * cellSize / rayDir

    tile = tile - torch.tensor([1], dtype=torch.long)

    return tile, dTile, dt, ddt


def castRayVectorDirALL(grid, rayStart, rayDir, maxdist, flipped_y_axis=True):
    grid_shape = grid.size()
    grid_width = grid_shape[1]
    grid_height = grid_shape[0]

    if flipped_y_axis:
        rayStartX = rayStart[0]
        rayStartY = grid_height - rayStart[1]
        rayDirX = rayDir[0]
        rayDirY = -rayDir[1]
    else:
        rayStartX = rayStart[0]
        rayStartY = rayStart[1]
        rayDirX = rayDir[0]
        rayDirY = rayDir[1]

    cellSize = torch.tensor([1.], dtype=torch.float)
    tileX, dtileX, dtX, ddtX = getHelpers(cellSize, rayStartX, rayDirX)
    tileY, dtileY, dtY, ddtY = getHelpers(cellSize, rayStartY, rayDirY)
    t = torch.tensor([0.], dtype=torch.float)

    t_out = torch.tensor([], dtype=torch.float)

    tileY_out = []
    tileX_out = []
    gridValues = []
    if dtX == 0:
        while tileX >= 0 and tileX <= grid_width - 1 and tileY >= 0 and tileY <= grid_height - 1:
            if t > maxdist:
                t = maxdist
                if grid[tileY, tileX] > torch.tensor([0.]):
                    t_out = torch.cat((t_out, t), dim=0)
                    tileY_out.append(tileY)
                    tileX_out.append(tileX)
                    gridValues.append(grid[tileY, tileX].item())
                break

            if grid[tileY, tileX] > torch.tensor([0.]):
                tileY_out.append(tileY)
                tileX_out.append(tileX)
                gridValues.append(grid[tileY, tileX].item())
                t_out = torch.cat((t_out, t), dim=0)  # <-- this works with pyro
                # but it does yield a wrong gradient
                # As a quick fix we use the euclidean distance instead
                # point1 = torch.vstack([rayStartX, rayStartY])
                # point2 = t * torch.vstack([rayDirX, rayDirY])
                # v = point2 - point1
                # # v = torch.vstack([tileX, tileY]) - torch.vstack([rayStartX, rayStartY])
                # dist = torch.linalg.norm(v)
                # print("dist: " + str(dist) + "  t: " + str(t))
                # t_out = torch.cat((t_out, dist.view(1)), dim=0)

            tileY = tileY + dtileY
            dt = dtY
            t = t + dt
            dtY = dtY + ddtY - dt
    elif dtY == 0:
        while tileX >= 0 and tileX <= grid_width - 1 and tileY >= 0 and tileY <= grid_height - 1:
            if t > maxdist:
                t = maxdist
                if grid[tileY, tileX] > torch.tensor([0.]):
                    t_out = torch.cat((t_out, t), dim=0)
                    tileY_out.append(tileY)
                    tileX_out.append(tileX)
                    gridValues.append(grid[tileY, tileX].item())
                break

            if grid[tileY, tileX] > torch.tensor([0.]):
                tileY_out.append(tileY)
                tileX_out.append(tileX)
                gridValues.append(grid[tileY, tileX].item())
                t_out = torch.cat((t_out, t), dim=0)  # <-- this works with pyro
                # but it does yield a wrong gradient
                # As a quick fix we use the euclidean distance instead
                # point1 = torch.vstack([rayStartX, rayStartY])
                # point2 = t * torch.vstack([rayDirX, rayDirY])
                # v = point2 - point1
                # # v = torch.vstack([tileX, tileY]) - torch.vstack([rayStartX, rayStartY])
                # dist = torch.linalg.norm(v)
                # print("dist: " + str(dist) + "  t: " + str(t))
                # t_out = torch.cat((t_out, dist.view(1)), dim=0)

            tileX = tileX + dtileX
            dt = dtX
            t = t + dt
            dtX = dtX + ddtX - dt
    else:
        while tileX >= 0 and tileX <= grid_width - 1 and tileY >= 0 and tileY <= grid_height - 1:
            if t > maxdist:
                t = maxdist
                if grid[tileY, tileX] > torch.tensor([0.]):
                    tileY_out.append(tileY)
                    tileX_out.append(tileX)
                    gridValues.append(grid[tileY, tileX].item())
                    t_out = torch.cat((t_out, t), dim=0)
                break

            if grid[tileY, tileX] > torch.tensor([0.]):
                tileY_out.append(tileY)
                tileX_out.append(tileX)
                gridValues.append(grid[tileY, tileX].item())
                t_out = torch.cat((t_out, t), dim=0)  # <-- this works with pyro
                # but it does yield a wrong gradient
                # As a quick fix we use the euclidean distance instead
                # point1 = torch.vstack([rayStartX, rayStartY])
                # point2 = t * torch.vstack([rayDirX, rayDirY])
                # v = point2 - point1
                # # v = torch.vstack([tileX, tileY]) - torch.vstack([rayStartX, rayStartY])
                # dist = torch.linalg.norm(v)
                # print("dist: " + str(dist) + "  t: " + str(t))
                # t_out = torch.cat((t_out, dist.view(1)), dim=0)

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

    tileY_out = torch.FloatTensor(tileY_out).long()
    tileX_out = torch.FloatTensor(tileX_out).long()
    gridValues = torch.FloatTensor(gridValues)
    return t_out, tileY_out, tileX_out, gridValues


def castRayAngleDirALL(grid, rayStart, angle, maxdist):
    rayDir = torch.stack((torch.cos(angle), torch.sin(angle)), 0)
    t, tileY, tileX, gridValues = castRayVectorDirALL(grid, rayStart, rayDir, maxdist)

    return t, tileY, tileX, gridValues

def lidar(map_grid, z_s, N_meas, maxdist):
    # return value of all non-zero cells together with the distance to that cell
    t = []
    gridValues = []
    tileX = []
    tileY = []
    angles = torch.arange(start=0, end=2*torch.pi, step=torch.pi*2/N_meas.item())
    #angles = 2*torch.pi*torch.rand(N_meas) # IMPORTANT TO CHANGE WHEN COMPARING over multiple runs!!!!
    for i in range(N_meas.item()):
        t_n, tileY_n, tileX_n, gridValues_n = castRayAngleDirALL(map_grid, z_s, angles[i], maxdist)
        t.append(t_n)
        gridValues.append(gridValues_n)
        tileX.append(tileX_n)
        tileY.append(tileY_n)
        
    tileX = torch.cat(tileX, 0)
    tileY = torch.cat(tileY, 0)
    return t, gridValues, angles, tileX, tileY