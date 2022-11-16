import pip

pyspin_whl = 'spinnaker_python-2.7.0.128-cp38-cp38-win_amd64.whl'

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

# Example
if __name__ == '__main__':
    install(pyspin_whl)