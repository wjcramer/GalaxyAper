# GalaxyAper
This code is designed to do resolved, matched mutli-filter aperture photometry on crowded fields of galaxies, particularly for galaxy clusters. Photutils is used for galaxy identification and de-blending, then initial galaxy apertures are applied in a user-specified filter. Extracted galaxies are then matched with an external catalogue in order to select galaxies within a given redshift. Finally, selected galaxies are passed through to the final routine, which does matched aperture photometry on multiple filter images of the same field. The same size and scale image should be used for each filter. 

The code was originally developed for use in high-redshift clusters (see https://arxiv.org/abs/2404.07355), so apertures are limited to r_50, r_90, and r_kron. If other apertures are needed it will be easy to modify the code upon request.

Installation:
Download package and cd into directory it is opened in:

```bash
python setup.py sdist bdist_wheel
pip install dist/galaxyaper-1.0.0.tar.gz
```

The example directory contains a script showing the workflow of this code. 
