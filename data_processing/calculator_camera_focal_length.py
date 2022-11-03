import numpy as np
import argparse


table_sensor_size_mm = {
    "1/4": {
        "width": 3.6,
        "height": 2.7,
        "diagonal": 4.5
    },
    "1/3": {
        "width": 4.8,
        "height": 3.6,
        "diagonal": 6.0
    },
    "1/2": {
        "width": 6.4,
        "height": 4.8,
        "diagonal": 8.0
    },
    "1/1.8": {
        "width": 7.1,
        "height": 5.4,
        "diagonal": 9.0
    },
    "2/3": {
        "width": 8.8,
        "height": 6.6,
        "diagonal": 11.0
    },
    "1": {
        "width": 12.8,
        "height": 9.6,
        "diagonal": 16
    }
}


def get_sensor_size(sensor_format: str):
    if sensor_format not in table_sensor_size_mm:
        raise Exception(f'Unknown format {sensor_format}')
    else:
        return table_sensor_size_mm[sensor_format]


FACTOR_MM2IN = 1.0 / 25.4


def in2mm(value: float):
    return value * 25.4


def mm2in(value: float):
    return value * FACTOR_MM2IN


def estimate_focal_length(
        working_distance: float, field_of_view: float, sensor_format: str, units: str = 'mm', **kwargs
):
    fov_horizontal = kwargs.get('fov_horizontal', True)
    if units not in ['mm', 'in']:
        raise Exception(f'Unknown unit: {units}')
    if units == 'in':
        working_distance = in2mm(working_distance)
        field_of_view = in2mm(field_of_view)
    s = 'width' if fov_horizontal else 'height'
    sensor = get_sensor_size(sensor_format)
    sensor_size = sensor[s]
    f = sensor_size * working_distance / (sensor_size + field_of_view)
    return np.round(f, 2)

if __name__ == '__main__':
    debug = False
    format_choices = list(table_sensor_size_mm)
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("-w", "--wd", help="the working distance", type=float, required=True, action="store")
    parser.add_argument("-f", "--fov", help='the field of view', type=float, required=True)
    parser.add_argument("-s", "--sensor", help="the sensor format", choices=format_choices, required=True)
    parser.add_argument(
        "-o", "--orientation", help="horizontal or vertical FOV", choices=['h', 'v'], default='h', required=False
    )
    parser.add_argument(
        "-u", "--units", help="units for the inputs", choices=['in', 'mm'], default='mm', required=False
    )
    args = parser.parse_args()
    if args.verbose:
        debug = True
    wd = args.wd
    fov = args.fov
    sensor = args.sensor
    orientation = args.orientation
    units = args.units
    fov_horizontal = orientation == 'h'
    f = estimate_focal_length(
        working_distance=wd, field_of_view=fov, sensor_format=sensor, units=units, fov_horizontal=fov_horizontal
    )

    print(f'Focal length: {f:.2f} mm')



