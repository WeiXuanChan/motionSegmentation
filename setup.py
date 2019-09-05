from setuptools import setup
setup(
  name = 'medImgProc',         # How you named your package folder (MyLib)
  packages = ['medImgProc'],   # Chose the same as "name"
  version = '1.0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Medical Image Processing module for viewing and editing.',   # Give a short description about your library
  author = 'Wei Xuan Chan',                   # Type in your name
  author_email = 'w.x.chan1986@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/WeiXuanChan/medImgProc',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/WeiXuanChan/medImgProc/archive/v1.0.0.tar.gz',    # I explain this later on
  keywords = ['medical', 'image'],   # Keywords that define your package best
  install_requires=['numpy','matplotlib','imageio','scipy','multiprocessing','pickle','inspect','trimesh','SimpleITK','re','pywt'],
  classifiers=[
    'Development Status :: 5 - Production/Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package    
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',    
    'License :: OSI Approved :: MIT License',   # Again, pick a license    
    'Programming Language :: Python :: 3.6',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.7',
  ],
)
