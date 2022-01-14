import torch
import pyro
from pyro.contrib.autoname import scope
import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistributionMixin

from misc.torch_truncnorm.TruncatedNormal_Modified import TruncatedNormal

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # 3.1415927410125732

import raycasting.raycasting_numba as raycasting

# class TruncNormal(TruncatedNormal.TruncatedNormal, TorchDistributionMixin):
class TruncNormal(TruncatedNormal, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TruncNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.a = self.a.expand(batch_shape)
        new.b = self.b.expand(batch_shape)
        super(TruncNormal, new).__init__(new.loc, new.scale, new.a, new.b)
        new._validate_args = self._validate_args
        return new

class lidarSubsampler():
    N_beams = None
    plate_context = None
    indexes = None
    counter = 0
    position = None

    ray_samples = None

    def __init__(self):
        pass

    def generate_plate_context(suffix="", Count=False):
        lidarSubsampler.plate_context = lidarSubsampler.indexes
        # lidarSubsampler.plate_context = pyro.plate("map_lidar_beams_{}_".format(lidarSubsampler.counter)+suffix, lidarSubsampler.N_beams, subsample=lidarSubsampler.indexes)
        if Count:
            lidarSubsampler.counter = lidarSubsampler.counter + 1

    def sample_all():
        lidarSubsampler.indexes = range(lidarSubsampler.N_beams)

    def sample_specific(indexes):
        # indexes: indexes to use
        lidarSubsampler.indexes = indexes

    def sample_random(N_samples):
        # N_samples: number of samples
        probs = torch.ones(lidarSubsampler.N_beams) / lidarSubsampler.N_beams
        lidarSubsampler.indexes = torch.multinomial(probs, N_samples).tolist()

    def do_ray_casting(position, map_grid_probabilities, lidarParams):
        lidarSubsampler.ray_samples = {}
        rayStart = position * lidarParams["meter2pixel"]
        #print(lidarSubsampler.plate_context)

        t, tileY, tileX, gridValues, angle = raycasting.castMultipleRaysAngleDirALL(map_grid_probabilities, rayStart, lidarSubsampler.plate_context, lidarSubsampler.N_beams, lidarParams["z_max"] * lidarParams["meter2pixel"])

        for i in range(len(lidarSubsampler.plate_context)):
            ray_samples = {}
            ray_samples["angle"] = angle[i]
            ray_samples["t"] = t[i]
            ray_samples["tileX"] = tileX[i]
            ray_samples["tileY"] = tileY[i]
            ray_samples["gridValues"] = gridValues[i]
            lidarSubsampler.ray_samples[str(lidarSubsampler.plate_context[i])] = ray_samples


def lidar_generate_labels(lidarParams, position, map_grid_probabilities, subsampling=None):
    lidarSubsampler.position = position
    lidarSubsampler.N_beams = lidarParams["N_lidar_beams"]

    if isinstance(subsampling, int):
        lidarSubsampler.sample_random(subsampling)
    elif type(subsampling) == list:
        lidarSubsampler.sample_specific(subsampling)
    else:  # assume that all should be sampled
        lidarSubsampler.sample_all()

    lidarSubsampler.generate_plate_context()
    lidarSubsampler.do_ray_casting(position, map_grid_probabilities, lidarParams)

    lidar_labels = []
    for n in lidarSubsampler.indexes:
        lidar_labels.append(str(n) + "/z_beam")
    return lidar_labels

def p_z_Map_beam_sampler(position, n, map_grid_probabilities, lidarParams):
    # Samples from the prior distribution over LTM
    # Notice that the the state sample, z_s, and sample from PB, z_PB, should not alter the sampling probability
    # of LTM, but should only be used to avoid sampling parts of the LTM that is independent of the current
    # situation for computational efficiency!
    # position: current state sample
    # z_PB: current sample from PB - in this case a tuble of an ray angle and sampled distance
    # LTM: variables storing current content of the LTM
    # Returns
    # z_LTM: a sample of the relevant LTM

    #rayStart = position * lidarParams["meter2pixel"]  # torch.stack((P_z_s_0_map_idx_x, P_z_s_0_map_idx_y))
    #t, tileY, tileX, gridValues = raycasting.castRayAngleDirALL(map_grid_probabilities, rayStart, angle, lidarParams["z_max"] * lidarParams["meter2pixel"])
    angle = lidarSubsampler.ray_samples[str(n)]["angle"]
    t = lidarSubsampler.ray_samples[str(n)]["t"]
    tileX = lidarSubsampler.ray_samples[str(n)]["tileX"]
    tileY = lidarSubsampler.ray_samples[str(n)]["tileY"]
    gridValues = lidarSubsampler.ray_samples[str(n)]["gridValues"]

    z_hit_xy_star = None
    if len(gridValues) != 0:
        for i in range(len(gridValues)):
            z_LTM_X_Y = pyro.sample('z_LTM_Map_{}_{}'.format(tileX[i], tileY[i]), dist.Bernoulli(gridValues[i]))
            if z_LTM_X_Y:
                # calculate intersection point
                z_hit_xy_star = torch.tensor([position[0] + torch.cos(angle)*(t[i]/ lidarParams["meter2pixel"]),
                                              position[1] + torch.sin(angle)*(t[i]/ lidarParams["meter2pixel"])])

                # Instead we could have returned:
                # z_beam_star = t[0] / lidarParams["meter2pixel"]
                # however calculating the gradient through the raytracing algo is inefficient

                break

    return z_hit_xy_star


def p_z_Map_prior(map_grid_probabilities, lidarParams):
    # we could sample all the cells in the current belief of the map,
    # however that would be very inefficient. Therefore, we instead
    # only sample cells intersecting with the directions of the lidars
    # laser beams, and save the distance coresponding to the first cell
    # that is sampled to be occupied in those directions.
    # arg:

    z_Map = {}  # dict to contain the LTM samples related to the map

    for n in lidarSubsampler.plate_context:  # we only need to sample the map cells that the lidar beams pass through
        with scope(prefix=str(n)):
            # angle = torch.tensor(n * (torch.pi * 2 / lidarSubsampler.N_beams))
            z_Map["z_hit_xy_star_" + str(n)] = p_z_Map_beam_sampler(lidarSubsampler.position, n, map_grid_probabilities, lidarParams)
    return z_Map


def p_z_beam_prior(lidarParams):
    # the following assumes that there always is an object in the range [0,z_max]!
    # consider if there should be added some likelyhood to not sense an object within this range,
    # e.g. by adding a z_max part as in "signel_beam_forward"

    beam_max_range = lidarParams["z_max"]

    # P_hit = 0.99999
    # P_rand = 0.00001
    # P_max = 0.0
    # P_short = 0.0 # not implemented yet! => should always be zero

    # assignment_probs = torch.tensor([P_hit, P_rand])
    # z_beam_category = pyro.sample('z_beam_category', dist.Categorical(assignment_probs), infer={"enumerate": "sequential"})

    # The uniform distribution in pytorch is on the interval [a,b), thus excluding b
    # leading to an incosistency of likelyhood at z_k=z_max if b=z_max, since using the uniform distribution
    # on the interval [0,z_max) would be the same as saying that the likelyhood of the sensor generating a
    # random measurement at z_k=z_max is zero.
    # to make sure that z_max is included in the interval, flip the distribution!
    p_z_beam = dist.Uniform(0.0, beam_max_range)
    z_beam = pyro.sample('z_beam', p_z_beam)

    return z_beam


def p_z_beam_posterior(z_beam_star, lidarParams):
    # made according to the book "Probabilistic Robotics"
    # z_k: value to evaluate the likelyhood of
    # z_beam_star: the "true" distance found be raytracing in the (sampled) map
    # beam_max_range: the max measurement distance of the beam sensor

    # Model params:
    beam_max_range = lidarParams["z_max"]
    sigma_hit = lidarParams["sigma_hit"]
    lambda_short = lidarParams["lambda_short"]
    P_hit = lidarParams["P_hit"]
    P_rand = lidarParams["P_rand"]
    P_max = lidarParams["P_max"]
    P_short = lidarParams["P_short"]  # not implemented yet! => should always be zero

    if z_beam_star is None:
        z_beam_star_ = beam_max_range
        P_rand = P_rand + P_hit
        # P_max = P_max + P_hit
        P_hit = 0.0
    else:
        z_beam_star_ = z_beam_star  # internal variable

    assignment_probs = torch.tensor([P_hit, P_rand, P_max, P_short])
    z_beam_category = pyro.sample('z_beam_category', dist.Categorical(assignment_probs), infer={"enumerate": "sequential"})

    if z_beam_category == 0:  # hit
        p_z_beam = TruncNormal(z_beam_star_, sigma_hit, 0.0, beam_max_range)

    elif z_beam_category == 1:  # random
        p_z_beam = dist.Uniform(0.0, beam_max_range)

    elif z_beam_category == 2:  # max
        p_z_beam = dist.Delta(beam_max_range)

    else:  # z_beam_category == 1: # short - TruncExp not implemented yet
        raise ValueError('TruncExp not implemented')
        # p_z_beam = TruncExp(rate=lambda_short, 0, beam_max_range)

    z_beam = pyro.sample('z_beam', p_z_beam)

    return z_beam


def p_z_Lidar_prior(lidarParams):

    z_lidar = {}  # dict to contain the PB samples related to the lidar

    for n in lidarSubsampler.plate_context:
        with scope(prefix=str(n)):
            z_beam_n = p_z_beam_prior(lidarParams)
            z_lidar["z_beam_" + str(n)] = z_beam_n

    return z_lidar


def p_z_Lidar_posterior(position, z_Map, lidarParams):
    z_lidar = {}  # dict to contain the PB samples related to the lidar

    for n in lidarSubsampler.plate_context:
        with scope(prefix=str(n)):
            z_hit_xy_star_n = z_Map["z_hit_xy_star_" + str(n)]

            if z_hit_xy_star_n is None:
                z_beam_n_star = lidarParams["z_max"]
            else:
                # calculate Eucledian Distance to intersection with occupied cell
                z_beam_n_star = torch.sqrt(torch.sum((position-z_hit_xy_star_n)**2))
                if z_beam_n_star == 0:  # the truncated normal in p_z_beam_posterior(...) does not support 0.0
                    z_beam_n_star = torch.finfo(torch.float32).eps

            z_beam_n = p_z_beam_posterior(z_beam_n_star, lidarParams)
            z_lidar["z_beam_" + str(n)] = z_beam_n

    return z_lidar
