import logging
import json
import os
import sys

sys.path.append('../')
import data_processing.confidence as cf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import skimage.filters as filters
from skimage.io import imread, imsave
import itertools
from scipy.optimize import least_squares

from skimage.util import img_as_ubyte
import matplotlib.ticker as ticker



base_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\test\BEAM_PROFILER_REALIGNED_LASER'
base_path = r'G:\Shared drives\ARPA-E Project\Lab\Data\Laser Tests\CAMERA\BEAM_PROFILING_20221212'
center = np.array([590, 593])
center = np.array([500, 500])
# pixel_size = 207.2227  # pixels/cm
pixel_size = 209.9845  # pixels/cm

def file_list(path='./', extension: str = ".txt"):
    files = []
    for file in os.listdir(path):
        if file.endswith(extension) and file.startswith('LCT'):
            files.append(file)
    return files


def gaussian_beam(r: np.ndarray, beam_radius: float, beam_power: float):
    """
    Estimates the gaussian profile of a laser.

    Parameters
    ----------
    r:np.ndarray
      The position in cm
    beam_radius:float
      The diameter of the beam in cm^2
    beam_power:float
      The power of the beam in W
    Returns
    -------
    np.ndarray:
      The intensity profile of the gaussian beam
    """
    r2 = np.power(r, 2.0)
    wz_sq = beam_radius ** 2.0
    intensity = (beam_power / (0.5 * np.pi * wz_sq)) * np.exp(
        -2.0 * r2 / wz_sq
    )

    return intensity


def model(xyc, b):
    # c = xyc[2]
    c = np.array([b[2], b[3]])
    xy = xyc[0]
    xy_shape = xyc[1]
    r = np.linalg.norm(xy - c, axis=1)
    beam_profile = gaussian_beam(r, beam_radius=b[0], beam_power=b[1]).reshape(xy_shape)
    beam_profile *= 255 / beam_profile.max()
    beam_profile -= beam_profile.min()
    return beam_profile


def fobj(b, xyc, y_exp):
    return (model(xyc, b) - y_exp).flatten()


def fobj_log(b, xyc, y_exp):
    new_b = [b[0], 10.0 ** b[1], b[2], b[3]]
    return (model(xyc, new_b) - y_exp).flatten()


def main(logger: logging.Logger):
    logger.info(f'Fitting gaussian beam')
    # Get the list of images
    image_list = file_list(path=base_path, extension='.jpg')
    im_laser = imread(os.path.join(base_path, image_list[0]))
    im_laser = np.zeros_like(im_laser, dtype=np.float64)
    # pattern = re.compile('\d+\.?\d?[eE]\d+')
    n_points = len(image_list)
    for im in image_list:
        # Get the exposure time from the csv
        # m = pattern.findall(im)
        # exposure_times.append(float(m[0]))
        img = imread(os.path.join(base_path, im)).astype(np.float64)
        # exposure_s = 1E6 / row['Exposure (us)']
        im_laser += img

    im_laser /= n_points

    im_max = im_laser.flatten().max()
    im_laser = 255 * im_laser / im_max
    im_laser -= im_laser.flatten().min()
    image = img_as_ubyte(im_laser.astype(int))

    imsave(os.path.join(base_path, 'average_img.jpg'), image)

    blurred = filters.gaussian(
        image, sigma=0.5)  # , mode='reflect')
    blurred = 255 * (blurred / blurred.flatten().max())
    # center = np.unravel_index(blurred.argmax(), blurred.shape)
    # center = np.array(center, dtype=float)
    logger.info(f'Initial position of the center: {center}')
    im_z_lim = [im_laser.flatten().min(), im_laser.flatten().max()]
    logger.info(f'Grayscale limits for the image: {im_z_lim}')

    x = np.arange(0, im_laser.shape[0]).astype(float)
    y = np.arange(0, im_laser.shape[1]).astype(float)

    xy = np.array([v for v in itertools.product(x, y)])
    # xyc = [xy, im_laser.shape, center]
    xyc = [xy, im_laser.shape]
    n = np.prod(im_laser.shape)

    b0 = [200, im_laser.flatten().max() * 0.7, center[0], center[1]]
    b0 = [b0[0], np.log10(b0[1]), b0[2], b0[3]]
    # b0 = [500, im_laser.flatten().max()*0.1]
    all_tol = (np.finfo(np.float64).eps) #** (1.0 / 2.0)
    res = least_squares(
        fobj, b0,
        loss='soft_l1', f_scale=0.1,
        jac='3-point',
        args=(xyc, im_laser.astype(float)),
        method='trf',
        # bounds=([20, 1], [500, im_laser.flatten().max()*1.5]),#, im_laser.shape[0], im_laser.shape[1]]),
        bounds=([5, 1E-2, 1E-2, 1E-2], [max(im_laser.shape), 255, im_laser.shape[0], im_laser.shape[1]]),
        # bounds=(
        # [10, -1, 0.01, 0.01], [max(im_laser.shape), np.log10(255), im_laser.shape[1], im_laser.shape[0]]),
        xtol=all_tol,
        ftol=all_tol,
        gtol=all_tol,
        max_nfev=1000 * n,
        x_scale='jac',
        verbose=2
    )

    popt = res.x
    logger.info(f'popt: {popt}')
    wz = popt[0]
    intensity = popt[1]
    cx = popt[2]
    cy = popt[3]
    fitted_center = [cx, cy]
    # fitted_center = center
    pcov = cf.get_pcov(res)
    # popt_log = np.array([popt[0], 10.0 ** popt[1], popt[2], popt[3]])
    # pcov_log = pcov
    # pcov_log[1, 1] = 10 ** pcov_log[1, 1]
    # pcov_log[2, 2] = 10 ** pcov_log[2, 2]
    # pcov_log[3, 3] = 10 ** pcov_log[3, 3]
    # ci = cf.confint(n=n, pars=popt_log, pcov=pcov_log)
    ci = cf.confint(n=n, pars=popt, pcov=pcov)
    # ci[0:1,:] = np.power(10, ci[0:1,:])
    ypred = model(xyc, popt)
    ypred_lim = [ypred.min(), ypred.max()]
    logger.info(f'ypred limits: {ypred_lim}')

    im_pred = img_as_ubyte(ypred.astype(int))
    imsave(os.path.join(base_path, 'predicted_img.jpg'), im_pred)

    # ypred, lpb, upb = cf.predint(x=xyc, xd=xyc, yd=im_laser, func=model, res=res)
    logger.info(f'wz (px): {wz:.6f}, 95% CI:{ci[0]}')
    logger.info(f'intensity (bits): {intensity:.6f}, 95% CI:{ci[1]}')
    logger.info(f'cx: {cx:.3f}, 95% CI:[{ci[2, 0]:.5f} {ci[2, 1]:.5f}] px')
    logger.info(f'cy: {cy:.3f}, 95% CI:[{ci[3, 0]:.5f} {ci[3, 1]:.5f}] px')
    logger.info(f'ypred limits: {ypred_lim}')

    # """
    # From:
    # https://scikit-image.org/docs/stable/auto_examples/edges/plot_circular_elliptical_hough_transform.html
    # """
    # edges = canny(image, sigma=2, low_threshold=10, high_threshold=15)
    # plt.imshow(edges)
    # # Detect two radii
    # hough_radii = np.arange(100, 350, 2)
    # hough_res = hough_circle(edges, hough_radii)
    # # Select the most prominent 3 circles
    # accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
    #                                            total_num_peaks=1)
    # # Draw them
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    # image = color.gray2rgb(image)
    # for center_y, center_x, radius in zip(cy, cx, radii):
    #     circy, circx = circle_perimeter(center_y, center_x, radius,
    #                                     shape=image.shape)
    #     print(f'center_y: {center_y}, center_x: {center_x}, radius: {radius}')
    #     image[circy, circx] = (220, 20, 20)
    #
    # ax.imshow(image, cmap=plt.cm.gray)
    #
    # plt.show()
    with open('../plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)

    norm1 = plt.Normalize(vmin=0, vmax=255)

    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(4.0, 4.5), constrained_layout=True)
    axes[0].imshow(im_laser, interpolation='none', norm=norm1)
    # axes[0].pcolormesh(
    #     x, y, im_laser.T, cmap=plt.cm.viridis, vmin=0, vmax=255,
    #     shading='gouraud', rasterized=True
    # )
    axes[1].imshow(ypred, interpolation='none', norm=norm1)
    # axes[1].pcolormesh(
    #     x, y, ypred.T, cmap=plt.cm.viridis, vmin=0, vmax=255,
    #     shading='gouraud', rasterized=True
    # )
    circle1 = plt.Circle(fitted_center[::-1], wz, ec='w', fill=False, clip_on=False, ls=(0, (3, 1)), lw=0.75)
    circle2 = plt.Circle(fitted_center[::-1], wz, ec='w', fill=False, clip_on=False, ls=(0, (3, 1)), lw=0.75)
    circles = [circle1, circle2]
    # axins1.add_patch(circle1)

    for ax, circ in zip(axes, circles):
        ax.set_xlabel('x (px)')
        ax.set_ylabel('y (px)')
        ax.axvline(x=fitted_center[1], color='w', ls=':', lw=0.75, alpha=0.5)
        ax.axhline(y=fitted_center[0], color='w', ls=':', lw=0.75, alpha=0.5)
        ax.add_patch(circ)
    axes[0].set_title('Average beam profile', fontweight='regular')
    # axes[0].set_title('Blurred')

    wz_cm = wz / pixel_size
    wz_err_cm = np.mean(np.abs(ci[0, :] - wz)) / pixel_size
    wz_text = f'$w_{{\\mathrm{{z}}}}$: {wz:.0f} px\n'
    wz_text += f'Center: {cx:.0f}, {cy:.0f} px'
    px_text = f'Pixel size: {pixel_size:.4f} px/cm'
    title_text = f'Fit: $w_{{\\mathrm{{z}}}}$: {wz_cm:.3f} cm'  # ± {wz_err_cm:.4f} cm'


    axes[1].text(
        0.95, 0.05, wz_text, color='w',
        transform=axes[1].transAxes, va='bottom', ha='right',
        fontsize=9
    )

    axes[0].text(
        0.95, 0.05, px_text, color='w',
        transform=axes[0].transAxes, va='bottom', ha='right',
        fontsize=9
    )

    axes[1].set_title(title_text, fontweight='regular')

    for ax in axes:
        ax.set_xlim(0, 1440)
        ax.set_ylim(top=0, bottom=1080)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(280))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(270))

    fig.savefig(os.path.join(base_path, 'result.png'), dpi=600)

    # region = np.s_[np.argwhere(np), 5:50]
    # x, y, z = x[region], y[region], z[region]
    #
    # fig2, ax2 = plt.subplots(subplot_kw=dict(projection='3d'))
    #
    # ls = LightSource(270, 45)
    # # To use a custom hillshading mode, override the built-in shading and pass
    # # in the rgb colors of the shaded surface calculated from "shade".
    # rgb = ls.shade(im_laser.T, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    # surf = ax2.plot_surface(x, y, im_laser, rstride=1, cstride=1, facecolors=rgb,
    #                        linewidth=0, antialiased=False, shade=False)

    plt.show()


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    log_file = os.path.join(base_path, 'results.log')
    ch = logging.StreamHandler()
    fh = logging.FileHandler(log_file)
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setLevel(logging.DEBUG)
    fh.setLevel(logging.DEBUG)
    # ch.setFormatter(c_format)
    fh.setFormatter(f_format)
    logger.addHandler(ch)
    logger.addHandler(fh)
    main(logger)
