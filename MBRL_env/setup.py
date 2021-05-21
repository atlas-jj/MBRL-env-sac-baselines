from setuptools import setup

setup(
    name='MBRL_env',
    version='0.1',
    keywords='',
    url='',
    packages=['dmbrl', 'mbbl'],
    install_requires=[
        'numpy==1.19.5',
        'scipy==1.5.4',
        'gym==0.17.3',
        'mujoco-py==1.50.1.68',
        'hydra-core'
    ]
)
