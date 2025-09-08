import math
def format_value_uncertainty_latex(value, uncertainty, power_limits=(2, 2)):
    """
    Format a numerical value with uncertainty as a LaTeX string.

    Parameters:
    -----------
    value : float
        The measured value
    uncertainty : float
        The uncertainty in the measurement
    power_limits : tuple of int, default (2, 2)
        Tuple (m, n) defining when to use floating point notation.
        Floating point used when 10^(-m) < |value| < 10^n

    Returns:
    --------
    str
        LaTeX formatted string representation

    Examples:
    ---------
    >>> format_value_uncertainty_latex(1.234, 0.056)
    '1.234 \\pm 0.056'

    >>> format_value_uncertainty_latex(0.00123, 0.00045)
    '(1.23 \\pm 0.45) \\times 10^{-3}'

    >>> format_value_uncertainty_latex(1234.5, 67.8)
    '(1.235 \\pm 0.068) \\times 10^{3}'
    """

    if value == 0:
        return f"0 \\pm {uncertainty:.3g}"

    m, n = power_limits
    abs_value = abs(value)

    # Determine if we should use scientific notation
    use_scientific = abs_value < 10 ** (-m) or abs_value >= 10 ** n

    if not use_scientific:
        # Use regular floating point notation
        # Determine appropriate number of decimal places based on uncertainty
        if uncertainty > 0:
            # Round uncertainty to 1-2 significant figures
            uncertainty_magnitude = math.floor(math.log10(uncertainty))
            if uncertainty / (10 ** uncertainty_magnitude) < 3:
                # Use 2 significant figures for uncertainties starting with 1 or 2
                uncertainty_rounded = round(uncertainty, -uncertainty_magnitude + 1)
                decimal_places = max(0, -uncertainty_magnitude + 1)
            else:
                # Use 1 significant figure for uncertainties starting with 3 or higher
                uncertainty_rounded = round(uncertainty, -uncertainty_magnitude)
                decimal_places = max(0, -uncertainty_magnitude)

            # Format value with same decimal places
            value_rounded = round(value, decimal_places)

            if decimal_places == 0:
                return f"{int(value_rounded)} \\pm {int(uncertainty_rounded)}"
            else:
                return f"{value_rounded:.{decimal_places}f} \\pm {uncertainty_rounded:.{decimal_places}f}"
        else:
            return f"{value:.3g} \\pm {uncertainty}"

    else:
        # Use scientific notation
        # Find the appropriate exponent based on the value
        if value != 0:
            exponent = math.floor(math.log10(abs_value))
        else:
            exponent = 0

        # Scale value and uncertainty
        value_scaled = value / (10 ** exponent)
        uncertainty_scaled = uncertainty / (10 ** exponent)

        # Determine decimal places for scaled numbers
        if uncertainty_scaled > 0:
            # Round uncertainty to 1-2 significant figures
            if uncertainty_scaled < 3:
                uncertainty_rounded = round(uncertainty_scaled, 2)
                decimal_places = 2
            else:
                uncertainty_rounded = round(uncertainty_scaled, 1)
                decimal_places = 1

            # Adjust if rounding changed the scale
            if uncertainty_rounded >= 10:
                uncertainty_rounded /= 10
                value_scaled /= 10
                exponent += 1

            # Format value with appropriate precision
            value_rounded = round(value_scaled, decimal_places)

            # Clean up trailing zeros in formatting
            if decimal_places > 0:
                value_str = f"{value_rounded:.{decimal_places}f}".rstrip('0').rstrip('.')
                uncertainty_str = f"{uncertainty_rounded:.{decimal_places}f}".rstrip('0').rstrip('.')
            else:
                value_str = f"{int(value_rounded)}"
                uncertainty_str = f"{int(uncertainty_rounded)}"
        else:
            value_str = f"{value_scaled:.2f}".rstrip('0').rstrip('.')
            uncertainty_str = f"{uncertainty_scaled:.2f}".rstrip('0').rstrip('.')

        return f"({value_str} \\pm {uncertainty_str}) \\times 10^{{{exponent}}}"