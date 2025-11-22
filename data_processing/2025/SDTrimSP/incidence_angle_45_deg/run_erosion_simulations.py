import os
import subprocess
from string import Template
from pathlib import Path


def run_sdtrimsp_simulation(e0, template_file='tri.inp.txt'):
    """
    Run SDTrimSP simulation with specified incident ion energy.

    Parameters:
    -----------
    e0 : float or int
        Incident ion energy value to substitute in the template
    template_file : str, optional
        Path to the template file (default: 'tri.inp.txt')

    Returns:
    --------
    dict
        Dictionary containing:
        - 'success': bool indicating if simulation ran successfully
        - 'output_dir': str path to the output directory
        - 'stdout': str standard output from SDTrimSP
        - 'stderr': str standard error from SDTrimSP
        - 'return_code': int return code from the process
    """

    try:
        # Read the template file
        with open(template_file, 'r') as f:
            template_content = f.read()

        # Create Template object and substitute the energy value
        # Ensure the energy value is formatted as a float
        e0_float = float(e0)
        template = Template(template_content)
        substituted_content = template.substitute(e0=f"{e0_float:.1f}")

        # Create the output directory path
        output_dir = Path(f'./simulation_results/{e0}')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write the tri.inp file in the output directory
        output_file = output_dir / 'tri.inp'
        with open(output_file, 'w') as f:
            f.write(substituted_content)

        print(f"Created input file: {output_file}")

        # Change to the output directory and run SDTrimSP
        original_cwd = os.getcwd()

        try:
            os.chdir(output_dir)
            print(f"Running SDTrimSP in directory: {output_dir.absolute()}")

            # Run the SDTrimSP command
            result = subprocess.run(
                ['SDTrimSP', 'tri'],
                capture_output=True,
                text=True,
                timeout=108000  # 5 minute timeout - adjust as needed
            )

            return {
                'success': result.returncode == 0,
                'output_dir': str(output_dir.absolute()),
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }

        finally:
            # Always return to original directory
            os.chdir(original_cwd)

    except FileNotFoundError as e:
        if 'tri.inp.txt' in str(e):
            return {
                'success': False,
                'output_dir': None,
                'stdout': '',
                'stderr': f'Template file {template_file} not found',
                'return_code': -1
            }
        else:
            return {
                'success': False,
                'output_dir': str(output_dir.absolute()) if 'output_dir' in locals() else None,
                'stdout': '',
                'stderr': 'SDTrimSP command not found. Make sure it\'s installed and in your PATH.',
                'return_code': -1
            }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'output_dir': str(output_dir.absolute()),
            'stdout': '',
            'stderr': 'SDTrimSP simulation timed out after 5 minutes',
            'return_code': -2
        }

    except Exception as e:
        return {
            'success': False,
            'output_dir': str(output_dir.absolute()) if 'output_dir' in locals() else None,
            'stdout': '',
            'stderr': f'Unexpected error: {str(e)}',
            'return_code': -3
        }


# Example usage:
if __name__ == "__main__":
    # Example: Run simulation with 1000 eV incident energy
    energy_values = [160]  # Multiple energies for batch processing

    for energy in energy_values:
        print(f"\n{'=' * 50}")
        print(f"Running simulation for E0 = {energy:.0f} eV")
        print(f"{'=' * 50}")

        result = run_sdtrimsp_simulation(energy)

        if result['success']:
            print(f"✅ Simulation completed successfully!")
            print(f"Output directory: {result['output_dir']}")
            if result['stdout']:
                print(f"SDTrimSP output:\n{result['stdout']}")
        else:
            print(f"❌ Simulation failed!")
            print(f"Error: {result['stderr']}")
            if result['output_dir']:
                print(f"Output directory: {result['output_dir']}")