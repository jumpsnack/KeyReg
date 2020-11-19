from easydict import EasyDict as edict

cfg = edict()

cfg.common = edict()
cfg.common.img_w = 261
cfg.common.img_h = 261
cfg.common.n_grid = 9
cfg.common.n_max = 3

cfg.augmentation = edict()
cfg.augmentation.use_green_channel = False
cfg.augmentation.augmentation_list = [
    'additive_gaussian_noise',
    'random_brightness',
    'no_aug']
cfg.augmentation.additive_gaussian_noise = edict()
cfg.augmentation.additive_gaussian_noise.std_min = 0
cfg.augmentation.additive_gaussian_noise.std_max = 1
cfg.augmentation.additive_speckle_noise = edict()
cfg.augmentation.additive_speckle_noise.intensity = 5
cfg.augmentation.motion_blur = edict()
cfg.augmentation.motion_blur.max_ksize = 15
cfg.augmentation.gamma_correction = edict()
cfg.augmentation.gamma_correction.min_gamma = 0.8
cfg.augmentation.gamma_correction.max_gamma = 1.2

cfg.homography = edict()
cfg.homography.scaling = edict()
cfg.homography.scaling.use_scaling = True
cfg.homography.scaling.min_scaling_x = 0.7
cfg.homography.scaling.max_scaling_x = 1.3
cfg.homography.scaling.min_scaling_y = 0.7
cfg.homography.scaling.max_scaling_y = 1.3
cfg.homography.perspective = edict()
cfg.homography.perspective.use_perspective = True
cfg.homography.perspective.min_perspective_x = 0.000001
cfg.homography.perspective.max_perspective_x = 0.0005
cfg.homography.perspective.min_perspective_y = 0.000001
cfg.homography.perspective.max_perspective_y = 0.0005
cfg.homography.translation = edict()
cfg.homography.translation.use_translation = True
cfg.homography.translation.max_horizontal_dis = 20
cfg.homography.translation.max_vertical_dis = 20
cfg.homography.shearing = edict()
cfg.homography.shearing.use_shearing = True
cfg.homography.shearing.min_shearing_x = -0.2
cfg.homography.shearing.max_shearing_x = 0.2
cfg.homography.shearing.min_shearing_y = -0.2
cfg.homography.shearing.max_shearing_y = 0.2
cfg.homography.rotation = edict()
cfg.homography.rotation.use_rotation = True
cfg.homography.rotation.max_angle = 30
