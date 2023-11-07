from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gentext',
    version='1.0.0',
    description='A Python library for text generation with RNNs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/lifewjola/gentext',
    author='Anjolaoluwa Ajayi (dataprincess)',
    author_email='anjolaajayi3@gmail.com',

    packages=find_packages(exclude=('tests', 'docs')),

    install_requires=[
        'tensorflow>=2.0'
    ],
    
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='text-generation library',
)