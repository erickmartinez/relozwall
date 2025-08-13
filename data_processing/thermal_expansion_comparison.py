import numpy as np
import pandas as pd
import heat_flux_fourier as hff
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import json

data_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\data\firing_tests\POCO_GRAPHITE_SPHERES'

material_properties = {
    'Material': ['POCO graphite', 'Glassy Carbon'],
    'Thermal conductivity (W/cm-K)': [0.95, 6.6E-2],
    'CTE (10^{-6}/K)': [6.98, 2.79],
    'Specific heat (J/g-K)': [0.702, 0.714],
    'Density (g/cm^3)': [1.781, 1.38]
}

heat_load = 40.  # MW/m^2

z_probes = np.array([0., 0.05, 0.1])  # cm
surface_area = np.pi * 0.01 * 0.25  # cm^2

reflectance = 40.4
absorptivity = 1.0 - (reflectance / 100.0)


def load_plot_style():
    with open('plot_style.json', 'r') as file:
        json_file = json.load(file)
        plot_style = json_file['defaultPlotStyle']
    mpl.rcParams.update(plot_style)


def get_properties_df(material_props):
    df = pd.DataFrame(data=material_props)
    df['Thermal diffusivity (cm^2/s)'] = df.eval(
        '`Thermal conductivity (W/cm-K)` / `Density (g/cm^3)` / `Specific heat (J/g-K)`')
    return df


def simulate_temperature(t: np.ndarray, properties_df: pd.DataFrame, surface_heat_load: float, z: np.ndarray,
                         alpha: float, s_area: float):
    strain = np.empty(len(properties_df), dtype=float)
    power = np.empty_like(strain)
    u_comparison = {}
    for i, r in properties_df.iterrows():
        kappa_1 = r['Thermal diffusivity (cm^2/s)']
        k0 = r['Thermal conductivity (W/cm-K)']
        flux = surface_heat_load * alpha / k0
        u = hff.get_ut(
            x=z, diffusion_time=t, rod_length=0.1,
            diffusivity=kappa_1,  # b[1],
            emission_time=t.max(),
            flux=flux, T0=25.
        )
        u_comparison[r['Material']] = u
        cte = r['CTE (10^{-6}/K)']
        delta_temp = u[-1, 1] - 25.
        strain[i] = cte * delta_temp
        power[i] = r['Specific heat (J/g-K)'] * r['Density (g/cm^3)'] * u[-1, 1] / 0.1 / 100. / t.max()
    df = properties_df
    df['Strain (%)'] = 100. * strain
    df['Absorbed power (MW/m^2)'] = power
    return u_comparison, df


def main():
    global material_properties
    global absorptivity
    global heat_load
    global z_probes
    global surface_area
    material_properties_df = get_properties_df(material_props=material_properties)
    t = np.linspace(start=0., stop=0.2, num=500)
    surface_heat_load = heat_load * 100.
    u_comparison, material_properties_df = simulate_temperature(
        t=t, properties_df=material_properties_df, surface_heat_load=surface_heat_load, z=z_probes, alpha=absorptivity,
        s_area=surface_area
    )

    load_plot_style()
    fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True)
    fig.set_size_inches(4.0, 6.5)

    u_poco, u_glassy = u_comparison['POCO graphite'], u_comparison['Glassy Carbon']
    axes[0].plot(t, u_poco[:, 0], ls='-', c='C0', label='POCO')
    axes[0].plot(t, u_glassy[:, 0], ls='-', c='C1', label='Glassy')
    axes[0].set_title("Surface temperature")
    axes[0].legend(loc='best', frameon=True)

    axes[1].plot(t, u_poco[:, 1], ls='-', c='C0', label='POCO')
    axes[1].plot(t, u_glassy[:, 1], ls='-', c='C1', label='Glassy')
    axes[1].set_title("Middle temperature")
    axes[1].legend(loc='best', frameon=True)

    axes[2].plot(t, u_poco[:, 2], ls='-', c='C0', label='POCO')
    axes[2].plot(t, u_glassy[:, 2], ls='-', c='C1', label='Glassy')
    axes[2].set_title("Back temperature")
    axes[2].legend(loc='best', frameon=True)

    colors = ['C0', 'C1']

    for i, ax in enumerate(axes):
        if i > 1:
            ax.set_xlabel("Time [s]")
        ax.set_ylabel("Temperature [°C]")
        ax.set_ylim(0, 3500)
        ax.set_xlim(t.min(), t.max())
        ax.tick_params(axis='y', which='both', direction='in', labelright=True)
        ax.yaxis.set_ticks_position('both')
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(500))

        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    fig2, axes2 = plt.subplots(nrows=3, ncols=1, constrained_layout=True)
    fig2.set_size_inches(4.0, 6.5)

    for i, r in material_properties_df.iterrows():
        u = u_comparison[r['Material']]
        cte = r['CTE (10^{-6}/K)'] * 1E-6
        mean_temperature = np.mean(u, axis=1)
        dT = mean_temperature - 25.0
        dLdt = np.hstack([0, cte * dT[1::]/t[1::]])
        absorbed_heat = r['Specific heat (J/g-K)'] * r['Density (g/cm^3)'] * np.hstack([0, dT[1::]/t[1::]]) * 0.1 / 100.
        axes2[0].plot(t, 100. * cte * dT, label=r['Material'], c=colors[i])
        axes2[1].plot(t, 100. * dLdt, label=r['Material'], c=colors[i])
        axes2[2].plot(t, absorbed_heat, label=r['Material'], c=colors[i])


    for i, ax in enumerate(axes2):
        if i > 1:
            ax.set_xlabel("Time [s]")
        ax.set_xlim(t.min(), t.max())
        ax.tick_params(axis='y', which='both', direction='in', labelright=True)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
        ax.legend(loc='best', frameon=True)

    # axes2[0].yaxis.set_major_locator(ticker.MultipleLocator(1000))
    # axes2[0].yaxis.set_minor_locator(ticker.MultipleLocator(500))

    axes2[0].set_ylabel("Strain [%]")
    axes2[1].set_ylabel("Strain rate [%/s]")
    axes2[2].set_ylabel("Absorbed heat [MW/m$^{\mathregular{2}}$]")

    # axes2[1].set_yscale("log")
    # axes2[2].set_yscale("log")

    plt.show()


if __name__ == '__main__':
    main()
