import setuptools

setuptools.setup(name='keras-opt',
                 author='Pedro Marques',
                 author_email='pedro.r.marques@gmail.com',
                 description='Scipy keras optimizer',
                 url='https://github.com/pedro-r-marques/keras-opt',
                 packages=setuptools.find_packages(),
                 install_requires = [
                     'numpy', 'tensorflow-gpu', 'scipy', 'tqdm'
                 ],
                 python_requires='>=3.6',
                 version='0.0.1')
