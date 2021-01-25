from setuptools import setup
setup(
  name = 'motionSegmentation',         # How you named your package folder (MyLib)
  packages = ['motionSegmentation'],   # Chose the same as "name"
  version = '2.7.18',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Explicit spatio-temporal regularization of motion tracking.',   # Give a short description about your library
  author = 'Wei Xuan Chan',                   # Type in your name
  author_email = 'w.x.chan1986@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/WeiXuanChan/motionSegmentation',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/WeiXuanChan/motionSegmentation/archive/v2.7.18.tar.gz',    # I explain this later on
  keywords = ['explicit', 'motion', 'regularization'],   # Keywords that define your package best
  install_requires=['numpy','autoD','scipy','trimesh','medImgProc','nfft'],
  classifiers=[
    'Development Status :: 5 - Production/Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package    
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',    
    'License :: OSI Approved :: MIT License',   # Again, pick a license    
    'Programming Language :: Python :: 3.6',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.7',
  ],
)
