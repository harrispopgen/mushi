import setuptools
import versioneer


with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='mushi',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='William DeWitt',
    author_email='wsdewitt@gmail.com',
    description='🍄 [mu]tation [s]pectrum [h]istory [i]nference 🍄',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/harrispopgen/mushi',
    # packages=setuptools.find_packages(exclude=['tests', 'docs', 'docs']),
    packages=['mushi'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7.*',
    scripts=['bin/mushi'],
    install_requires=[
        'jax',
        'jaxlib',
        'prox-tv @ git+https://github.com/albarji/proxTV#egg=prox-tv',
        'seaborn',
        'pandas',
        'matplotlib',
        'numpy',
        'scipy'
    ],
)
