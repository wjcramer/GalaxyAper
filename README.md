# GalaxyAper
This code is designed to do resolved, matched mutli-filter aperture photometry on crowded fields of galaxies, particularly for galaxy clusters. Photutils is used for galaxy identification and de-blending, then initial galaxy apertures are applied in a user-specified filter. Extracted galaxies are then matched with an external catalogue in order to select galaxies within a given redshift, and append additional desired quantities, such as mass, to the final catalogue. Finally, selected galaxies are passed through to the final routine, which does matched aperture photometry on multiple filter images of the same field. The same size and scale image should be used for each filter. 

The code was originally developed for use in high-redshift clusters (see https://arxiv.org/abs/2404.07355), so apertures are limited to r_50, r_90, and r_kron, and has been tested only on HST data. If other apertures are needed it will be easy to modify the code upon request.

Installation:

```bash
pip install GalaxyAper
```

The example directory contains a script showing the workflow of this code. 
