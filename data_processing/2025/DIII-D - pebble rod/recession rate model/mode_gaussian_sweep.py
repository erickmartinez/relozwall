import numpy as np
import matplotlib.pyplot as plt
from data_processing.misc_utils.plot_style import load_plot_style

T_RANGE = [0, 3]  # s
SWEEP_RATE = 1E-2 # cm / s
SIGMA = 2 # cm
ANGLE = 1.5 # deg
NU_MODEL = 'LINEAR'
ROD_DIAMETER = 1.
INITIAL_HEIGHT = 0.2

def gaussian_pulse(x, t, w, s):
    return np.exp(-2*(x - w*t + 4)**2/(2*s**2)) + np.exp(-2*(x - w*t - 6)**2/(2*s**2))

def beam_interaction_area(r, h, theta):
    top = (np.pi * r ** 2) * np.cos(theta)
    sides =  np.pi * r * np.sin(theta) * h
    result = top * np.ones_like(h)
    msk = h > 0
    result[msk] = result[msk] + sides[msk]
    return result

def model(x, t, w, s, r, h, theta):
    return gaussian_pulse(x, t, w, s) * beam_interaction_area(r, h, theta)

def main(rod_diameter, initial_height, t_range, sweep_rate, sigma, angle, nu_model):
    plot_style = load_plot_style()
    t = np.linspace(t_range[0], t_range[1], num=200)
    x = np.linspace(-20*rod_diameter, 20*rod_diameter, num=200)
    omega = 2 * np.pi * sweep_rate
    theta = np.radians(90 - angle)
    if nu_model == 'LINEAR':
        nu = 0.075
        h = initial_height - nu * t

    current = model(x, t, omega, sigma, rod_diameter, h, theta)

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(4.5, 3)

    ax.plot(t, current)
    model_txt = r'\begin{align*}'
    model_txt += r'I(t) &= A_p(t) \left[ e^{-2(x - \omega t +3)^2 / (2\sigma^2)}'
    model_txt += r' + e^{-2(x - \omega t +4)^2 / (2\sigma^2)} \right]\\'
    model_txt += r'A_p(t) &= \pi r^2 \cos\theta + \pi r h(t) \sin\theta \\'
    model_txt += r'h(t) &= h_0 - \nu t'
    model_txt += r'\end{align*}'
    ax.text(
        0.025, 0.975, model_txt,
        transform = ax.transAxes, ha = 'left', va = 'top', fontsize = 10,
        color = 'k', usetex = True,
    )

    ax.set_xlabel('time (s)')
    ax.set_ylabel('current (A)')
    ax.set_xlim(t_range[0], t_range[1])
    ax.set_ylim(top=0.6)

    fig.savefig(f'./figures/mode_gaussian_sweep_{nu_model}.png', dpi=600)

    plt.show()

if __name__ == '__main__':
    main(
        rod_diameter=ROD_DIAMETER, initial_height=INITIAL_HEIGHT, t_range=T_RANGE,
        sweep_rate=SWEEP_RATE, sigma=SIGMA, angle=ANGLE, nu_model=NU_MODEL
    )


