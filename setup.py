try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
        'name': 'NeuroDemo',
        'description': 'Neural Network demonstration',
        'author': 'Mika Tammi',
        'url': 'https://github.com/mikatammi/neurodemo',
        'download_url': 'https://github.com/mikatammi/neurodemo',
        'author_email': 'mikatammi@gmail.com',
        'version': '0.1',
        'requires': ['nose',
                     'sklearn',
                     'sknn',
                     'numpy'],
        'scripts': ['neurodemo.py']
}

setup(**config)
