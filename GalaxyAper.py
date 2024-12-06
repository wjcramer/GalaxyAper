import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from astropy.io import fits
import astropy.io.ascii
from astropy.table import Table, vstack
from astropy.visualization import SqrtStretch, simple_norm
from astropy.convolution import convolve
from astropy.coordinates import Angle
from astropy.wcs import WCS
from astropy.modeling.models import Gaussian2D
from astropy.coordinates import SkyCoord
from astropy import units as u
from photutils.segmentation import SourceFinder
from photutils.segmentation import SourceCatalog
from photutils import segmentation, morphology
from photutils.aperture import EllipticalAperture
from photutils.datasets import make_gaussian_sources_image
from photutils.aperture import aperture_photometry
from photutils.aperture import aperture_photometry
from photutils.datasets import make_gaussian_sources_image
from photutils.background import Background2D, MedianBackground
from scipy.optimize import root_scalar
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse

# Optional Local Files

# Name of Readme file included with this software
README = 'README'

# Output Formatting
# Directory for output
OUTPUT = '.'
if not os.path.isdir(OUTPUT):
    os.mkdir(OUTPUT)
#

#Initialize Variables

KERNEL = 3
SIZE = 3
THRESH = 8
NPIX = 20
NLEV = 32
CONTRAST = 0.001
RMS = 0.004
VERBOSE = True
FILTER = 'F160W'
ID_COL = 'id'
RA_COL = 'ra'
DEC_COL = 'dec'
ZSPEC_COL = 'zspec'
ZPHOT_COL = 'zfast'
LMASS_COL = 'lmass'
Z_LOW = 1.53
Z_HIGH = 1.69

def convolve_data2(data, kernel=KERNEL, size=SIZE):
    """
    Convolves the input data with a 2D Gaussian kernel.

    Parameters:
    -----------
    data : numpy.ndarray
        The input data (e.g., image or 2D array) that will be convolved with the kernel.
    
    kernel : float, optional
        The full width at half maximum (FWHM) of the Gaussian kernel. This determines the spread of the kernel.
        Default is a value specified by the global variable `KERNEL`.
    
    size : int, optional
        The size of the Gaussian kernel (i.e., its dimensions). This will define the kernel's spatial extent.
        Default is a value specified by the global variable `SIZE`.
    
    Returns:
    --------
    convolved_data : numpy.ndarray
        The result of convolving the input data with the Gaussian kernel.
    """
    kernel = make_2dgaussian_kernel(kernel, size=size)  # FWHM = 3.
    convolved_data = convolve(data, kernel)
    return convolved_data

def sourceDetect(convolved_data, rms=RMS, thresh=THRESH, npix=NPIX, nlev=NLEV, contrast=CONTRAST):
    """
    Detects sources in the convolved data by creating a segmentation map, deblending sources,
    and saving the results to a FITS file and a CSV table.

    Parameters:
    -----------
    convolved_data : numpy.ndarray
        The input convolved data (e.g., a smoothed image) in which sources will be detected.
    
    rms : float, optional
        The root mean square (RMS) noise level in the image. Default is set by the global `RMS` value.
    
    thresh : float, optional
        The threshold multiplier for detecting sources. It is applied to the RMS to determine the detection threshold. Default is set by the global `THRESH` value.
    
    npix : int, optional
        The minimum number of pixels a detected source must span to be considered a valid detection. Default is set by the global `NPIX` value.
    
    nlev : int, optional
        The number of levels used for deblending sources. Default is set by the global `NLEV` value.
    
    contrast : float, optional
        The contrast parameter for deblending. Default is set by the global `CONTRAST` value.
    
    Returns:
    --------
    cat : SourceCatalog
        A source catalog object containing the detected sources and their properties.
    
    tbl : Table
        A table of source properties (e.g., centroids, flux) written to a CSV file.
    
    segm_deblend : numpy.ndarray
        The deblended segmentation map.
    """
    threshold = np.mean(rms)*thresh
    segment_map = detect_sources(convolved_data, threshold, npixels=npix)
    #Deblend segmentation map in similar way as source extractor
    segm_deblend = deblend_sources(convolved_data, segment_map,
                               npixels=npix, nlevels=nlev, contrast=contrast,
                               progress_bar=False)
    # Create a Primary HDU object
    segm = fits.PrimaryHDU(data=segm_deblend, header=header)
    # Create an HDUList
    segml = fits.HDUList([segm])
    # Write to a new FITS file
    output_filename = "segmentation_map.fits"
    segml.writeto(output_filename, overwrite=True)
    cat = SourceCatalog(convolved_data, segm_deblend, convolved_data=convolved_data)
    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'  # optional format
    tbl['ycentroid'].info.format = '.2f'
    tbl['kron_flux'].info.format = '.2f'
    astropy.io.ascii.write(tbl, 'Source_table.txt', format='csv', overwrite=True)
    return cat, tbl, segm_deblend
    
def photutil_source(data, kernel=KERNEL, rms=RMS, size=SIZE, thresh=THRESH, npix=NPIX, nlev=NLEV, contrast=CONTRAST):
    """
    Processes the input data by performing convolution and source detection. It returns the convolved data,
    the source catalog, a table of source properties, and the deblended segmentation map.

    Parameters:
    -----------
    data : str or numpy.ndarray
        The input data, either as a file path to a FITS file or as a numpy array containing the data (e.g., an image).
    
    kernel : float, optional
        The full width at half maximum (FWHM) of the Gaussian kernel. Used in the convolution. Default is set by `KERNEL`.
    
    rms : float, optional
        The root mean square (RMS) noise level in the image. Used for source detection. Default is set by `RMS`.
    
    size : int, optional
        The size of the Gaussian kernel (its spatial extent). Default is set by `SIZE`.
    
    thresh : float, optional
        The threshold multiplier for source detection. It is applied to the RMS to determine the detection threshold. Default is set by `THRESH`.
    
    npix : int, optional
        The minimum number of pixels that a detected source must span to be considered valid. Default is set by `NPIX`.
    
    nlev : int, optional
        The number of levels for deblending sources. Default is set by `NLEV`.
    
    contrast : float, optional
        The contrast parameter used for deblending sources. Default is set by `CONTRAST`.
    
    Returns:
    --------
    convolved_data : numpy.ndarray
        The smoothed or convolved data after applying the Gaussian kernel.
    
    cat : SourceCatalog
        The source catalog containing the detected sources and their properties.
    
    tbl : Table
        A table containing the properties of the detected sources, written to a CSV file.
    
    segm_deblend : numpy.ndarray
        The deblended segmentation map showing source locations and contours.
    """
    if isinstance(data, str):
        data = fits.getdata(data)
    elif not isinstance(data, np.ndarray):
        raise TypeError('data must be str or numpy.ndarray')
    convolved_data = convolve_data2(data, kernel=kernel, size=size)
    cat, tbl, segm_deblend = sourceDetect(convolved_data, thresh=thresh, npix=npix, nlev=nlev, contrast=contrast)
    
    
    return convolved_data, cat, tbl, segm_deblend

# define the objective function to minimize
def ellip_opt_fcn(radius, data, aperture, ba_ratio, normflux):
    aperture.a = radius
    aperture.b = radius * ba_ratio
    flux, _ = aperture.do_photometry(data)
    return flux[0] - normflux


# define the optimizer
def ellip_opt(data, ellip_aper, normflux, opt_fcn,
             min_radius=0.01, max_radius=100):
    ba_ratio = ellip_aper.b / ellip_aper.a
    args = (data, ellip_aper, ba_ratio, normflux)
    result = root_scalar(opt_fcn, args=args,
                         bracket=[min_radius, max_radius],
                         method=None)
    return result.root

def photometry_50_90(data, header, cat, tbl, segm_deblend, filter_name=FILTER, verbose=False):
    """
    Performs photometric measurements on sources detected in astronomical images, calculating fluxes within elliptical
    apertures at multiple radii (50% and 90% of the flux). It calculates the semi-major and semi-minor axes, the
    orientations, and outputs the results in a CSV file.

    Parameters:
    -----------
    data : str or numpy.ndarray
        The input data, either as a file path to a FITS file or a numpy array containing the image data.
    
    header : astropy.io.fits.Header
        The header of the FITS file used for WCS transformations to convert pixel coordinates to world coordinates (RA/Dec).
    
    cat : SourceCatalog
        The source catalog containing the detected sources' parameters, including Kron radius, flux, and other properties.
    
    tbl : Table
        The table containing the properties of detected sources, including centroid positions, orientation, and flux values.
    
    segm_deblend : numpy.ndarray
        The deblended segmentation map used to identify individual sources in the image.
    
    filter_name : str, optional
        The name of the filter used in the photometric observations. It is used to name output columns and files.
        Default is set by `FILTER`.
    
    verbose : bool, optional
        If True, prints status updates during the photometric measurements. Default is False.
    
    Returns:
    --------
    None
        The function writes the results to a CSV file, containing photometric data for each detected source.
    
    Notes:
    ------
    The CSV output includes:
        - Source ID
        - Pixel and world (RA/Dec) coordinates
        - Semi-major and semi-minor axes
        - Flux measurements at 50% and 90% radii
        - Orientation angle
    """
    semimaj = cat.kron_params[0] * cat.kron_radius.value * tbl['semimajor_sigma']
    semimin = cat.kron_params[0] * cat.kron_radius.value * tbl['semiminor_sigma']
    x = tbl['xcentroid']
    y = tbl['ycentroid']
    orientation = tbl['orientation']
    
    a_hl_list = np.zeros(semimaj.size)
    b_hl_list = np.zeros(semimaj.size)
    a_hl_list_90 = np.zeros(semimaj.size)
    b_hl_list_90 = np.zeros(semimaj.size)
    inner_flux = np.zeros(semimaj.size)
    wcs = WCS(hdul[0].header)
    data_csv = np.zeros((semimaj.size, 15))
    
    for i in range(0, semimaj.size):
        mask = np.zeros(segm_deblend.data.shape, dtype=bool)
        bad = np.where((segm_deblend.data != i + 1) | np.isnan(data))
        ok = np.where(segm_deblend.data == 0)
        mask[bad] = True
        mask[ok] = False
    
        xypos = (x[i], y[i])
        theta = orientation[i]
        aper = EllipticalAperture(xypos, a=semimaj[i].value, b=semimin[i].value, theta=theta)
        ba_ratio = aper.b / aper.a
    
        try:
            max_radius_value = semimaj[i].value * 1.04
            target_flux1 = tbl['kron_flux'][i] / 2.0  # e.g., half the flux
            a_hl = ellip_opt(data, aper, target_flux1, ellip_opt_fcn, max_radius=max_radius_value)
            b_hl = a_hl * ba_ratio
            a_hl, b_hl
        except ValueError as e:
            if str(e) == "f(a) and f(b) must have different signs":
                a_hl = 0
                b_hl = 0
            else:
                raise e
        a_hl_list[i] = a_hl
        b_hl_list[i] = b_hl
        if a_hl != 0:
            aper_hl = EllipticalAperture(xypos, a=a_hl, b=b_hl, theta=theta)
            aper_h2 = EllipticalAperture(xypos, a=semimaj[i].value, b=semimin[i].value, theta=theta)
        
            try:
                max_radius_value = semimaj[i].value * 1.04
                target_flux1 = tbl['kron_flux'][i] * 0.9  # e.g., 90% the flux
                a_hl_90 = ellip_opt(data, aper, target_flux1, ellip_opt_fcn, max_radius=max_radius_value)
                b_hl_90 = a_hl_90 * ba_ratio
                a_hl_90, b_hl_90
            except ValueError as e:
                if str(e) == "f(a) and f(b) must have different signs":
                    a_hl_90 = 0
                    b_hl_90 = 0
                else:
                    raise e
            a_hl_list_90[i] = a_hl_90
            b_hl_list_90[i] = b_hl_90
            if a_hl_90 != 0:
                aper_hl_90 = EllipticalAperture(xypos, a=a_hl_90, b=b_hl_90, theta=theta)
        
                phot_table = aperture_photometry(data, aper_hl, method='subpixel', mask=mask, subpixels=5)
                data_csv[i, 0] = int(phot_table['id'][0].item() + i)
                data_csv[i, 1] = phot_table['xcenter'][0].to_value()
                data_csv[i, 2] = phot_table['ycenter'][0].to_value()
                coords = wcs.pixel_to_world(x, y)
                data_csv[i, 3] = coords.ra.deg[i]
                data_csv[i, 4] = coords.dec.deg[i]
                data_csv[i, 5] = semimaj[i].value
                data_csv[i, 6] = semimin[i].value
                data_csv[i, 7] = a_hl_list[i]
                data_csv[i, 8] = b_hl_list[i]
                data_csv[i, 9] = a_hl_list_90[i]
                data_csv[i, 10] = b_hl_list_90[i]
                data_csv[i, 11] = orientation.value[i]
                data_csv[i, 12] = phot_table['aperture_sum'][0].item()
                phot_table = aperture_photometry(data, aper_h2, method='subpixel', mask=mask, subpixels=5)
                data_csv[i, 13] = phot_table['aperture_sum'][0].item()
                phot_table = aperture_photometry(data, aper_hl_90, method='subpixel', mask=mask, subpixels=5)
                data_csv[i, 14] = phot_table['aperture_sum'][0].item()
            if verbose:
                print(f"Processed galaxy number {i}")

    inner_col = f"{filter_name}_inner"
    outer_col = f"{filter_name}_outer"
    ninety_col = f"{filter_name}_90"
    
    header = ['id', 'xcenter', 'ycenter', 'xworld', 'yworld', 'semimajor', 'semiminor', 'a_hl',
              'b_hl', 'a_hl_90', 'b_hl_90', 'angle', inner_col, outer_col, ninety_col]

    for row in data_csv:
        row[0] = int(row[0])  # Ensure the first column is an integer

    output_name=f"{filter_name}_full_array_iterative.csv"

    with open(output_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # Write the header
        writer.writerow(header)
    
        # Write multiple rows
        writer.writerows(data_csv)


def GalaxyAper(data, header, kernel=KERNEL, rms=RMS, size=SIZE, thresh=THRESH, npix=NPIX, nlev=NLEV,
               contrast=CONTRAST, filter_name=FILTER, verbose=VERBOSE):
    """
    Performs source detection, convolution, and photometric measurements on astronomical images, including
    calculating fluxes within elliptical apertures. The function also generates a convolved image and outputs
    the photometric results to a CSV file.

    Parameters:
    -----------
    data : str or numpy.ndarray
        The input data, either as a file path to a FITS file or a numpy array containing the image data.
    
    header : astropy.io.fits.Header
        The header of the FITS file used for WCS transformations to convert pixel coordinates to world coordinates (RA/Dec).
    
    kernel : float, optional
        The kernel used for convolution (e.g., Full Width at Half Maximum, FWHM). Default is set by `KERNEL`.
    
    rms : float, optional
        The root mean square noise in the image. Default is set by `RMS`.
    
    size : int, optional
        The size of the convolution kernel. Default is set by `SIZE`.
    
    thresh : float, optional
        The threshold used for source detection. Default is set by `THRESH`.
    
    npix : int, optional
        The minimum number of pixels used for source detection. Default is set by `NPIX`.
    
    nlev : int, optional
        The number of levels used for deblending sources. Default is set by `NLEV`.
    
    contrast : float, optional
        The contrast parameter used for deblending sources. Default is set by `CONTRAST`.
    
    filter_name : str, optional
        The name of the filter used in the photometric observations. Default is set by `FILTER`.
    
    verbose : bool, optional
        If True, prints status updates during the process. Default is set by `VERBOSE`.

    Returns:
    --------
    None
        The function writes the photometric results to a CSV file, containing flux measurements and source properties.

    Notes:
    ------
    This function internally calls `photutil_source` for source detection and convolution, and `photometry_50_90`
    for calculating fluxes at different radii. The results are stored in a CSV file for further analysis.
    """
    if isinstance(data, str):
        data = fits.getdata(data)
    elif not isinstance(data, np.ndarray):
        raise TypeError('data must be str or numpy.ndarray')

    # Capture the outputs of photutil_source
    convolved_data, cat, tbl, segm_deblend = photutil_source(
        data, kernel=kernel, rms=rms, size=size, thresh=thresh, npix=npix, nlev=nlev, contrast=contrast
    )

    # Pass the outputs to photometry_50_90, include verbose flag
    photometry_50_90(data, header, cat=cat, tbl=tbl, segm_deblend=segm_deblend, verbose=verbose)

    


def srcor(x1in, y1in, x2in, y2in, dcr=1.5, magnitude=None, spherical=2, count=False, silent=False):
    """
    Match sources between two sets of coordinates (RA, DEC), using a critical distance (dcr) of 1.5 arcseconds.
    """
    coords1 = SkyCoord(ra=x1in * u.deg, dec=y1in * u.deg, frame='icrs')
    coords2 = SkyCoord(ra=x2in * u.deg, dec=y2in * u.deg, frame='icrs')
    # Perform the matching
    idx, sep, _ = coords1.match_to_catalog_sky(coords2)

    # Ensure idx is always an array, even if there is a single match
    if np.isscalar(idx):
        idx = np.array([idx])

    # Filter out matches that exceed the critical distance
    matched = sep.arcsec <= dcr
    ind1 = np.where(matched)[0]  # Indices of sources in the first list

    # If there is more than one match, apply the boolean mask
    if idx.size > 1:
        ind2 = idx[matched]  # Indices of matched sources in the second list
    else:
        # If only one match, use idx directly without applying the mask
        ind2 = idx

    # If no matches were found, return empty arrays
    if len(ind1) == 0:
        return np.array([]), np.array([])

    # Return the result
    if count:
        return len(ind1), ind1, ind2
    else:
        return ind1, ind2

def catalog_matching(catalog_file, GalaxyAper_output=None, id_col=ID_COL, ra_col=RA_COL, dec_col=DEC_COL, zspec_col=ZSPEC_COL,
                   zphot_col=ZPHOT_COL, lmass_col=LMASS_COL, z_low=Z_LOW, z_high=Z_HIGH, z_choice='zspec'):
    """
    Matches sources from a galaxy catalog with those detected in a photometric aperture catalog based on their
    sky positions (RA/Dec), and writes the matched catalog entries to a new output file. The function filters
    sources by redshift and computes the closest catalog match based on angular separation.

    Parameters:
    -----------
    catalog_file : str
        The path to the FITS catalog file containing the galaxy data (e.g., redshift, mass, RA/Dec).
    
    GalaxyAper_output : str, optional
        The path to the CSV file generated by the `GalaxyAper` function, containing aperture photometry results.
        Default is `None`, in which case it uses a default output filename derived from `FILTER`.
    
    id_col : str, optional
        The name of the ID column in the FITS catalog. Default is `ID_COL`.
    
    ra_col : str, optional
        The name of the Right Ascension column in the FITS catalog. Default is `RA_COL`.
    
    dec_col : str, optional
        The name of the Declination column in the FITS catalog. Default is `DEC_COL`.
    
    zspec_col : str, optional
        The name of the redshift (spectroscopic) column in the FITS catalog. Default is `ZSPEC_COL`.
    
    zphot_col : str, optional
        The name of the redshift (photometric) column in the FITS catalog. Default is `ZPHOT_COL`.
    
    lmass_col : str, optional
        The name of the stellar mass column in the FITS catalog. Default is `LMASS_COL`.
    
    z_low : float, optional
        The lower bound for redshift filtering. Default is `Z_LOW`.
    
    z_high : float, optional
        The upper bound for redshift filtering. Default is `Z_HIGH`.
    
    z_choice : str, optional
        The redshift source to use for filtering ('zspec', 'zphot', or 'zboth'). Default is 'zspec'. 'zboth' Considers anything
        between the specified redshift range (zphot or zspec)

    Returns:
    --------
    None
        The function writes the matched galaxy catalog to a new file (`z_member_catalogue.cat`) with photometric
        measurements, galaxy properties, and the closest matching catalog entries.

    Notes:
    ------
    This function utilizes `srcor` for matching sources between the catalog and photometric data by their
    RA/Dec positions. It outputs the matched entries, including flux measurements and galaxy properties, to a text
    file suitable for further analysis.
    """

    # Use the default catalog_file if not provided
    if GalaxyAper_output is None:
        GalaxyAper_output = f"{FILTER}_full_array_iterative.csv"
    
    print(f"Using GalaxyAper file: {GalaxyAper_output}")
    # Open the FITS file and read the table (assuming it's a binary table)
    fits_file = catalog_file
    hdulist = fits.open(fits_file)

    # Extract the data from the table
    data = hdulist[1].data  # This assumes the data is in the second HDU (index 1)

    # Extract the columns by index (adjusted to column names)
    catalog_id = data[id_col]
    ra_ref = data[ra_col]
    dec_ref = data[dec_col]
    zspec = data[zspec_col] if zspec_col in data.dtype.names else None
    zphot = data[zphot_col] if zphot_col in data.dtype.names else None
    lmass = data[lmass_col] if lmass_col in data.dtype.names else None

    hdulist.close()

    # Now read the CSV for additional data
    csv_data = np.genfromtxt(GalaxyAper_output, delimiter=',', skip_header=1)

    # Extract the columns from the CSV file
    id1 = csv_data[:, 0].astype(int)
    xcenter = csv_data[:, 1]
    ycenter = csv_data[:, 2]
    RA = csv_data[:, 3]
    DEC = csv_data[:, 4]
    semimaj = csv_data[:, 5]
    semimin = csv_data[:, 6]
    a_hl = csv_data[:, 7]
    b_hl = csv_data[:, 8]
    a_hl_90 = csv_data[:, 9]
    b_hl_90 = csv_data[:, 10]
    angle = csv_data[:, 11]
    F160W_inner = csv_data[:, 12]
    F160W_outer = csv_data[:, 13]
    F160W_90 = csv_data[:, 14]

    # Define critical distance (1.5 arcseconds)
    dcr = 1.5  # arcseconds

    # Filter data for zspec between z_low and z_high
    # Apply the selection based on z_choice
    if z_choice == 'zspec' and zspec is not None:
        good = np.where((zspec > z_low) & (zspec < z_high))[0]
    elif z_choice == 'zphot' and zphot is not None:
        good = np.where((zphot > z_low) & (zphot < z_high))[0]
    elif z_choice == 'zboth':
        good_zspec = np.where((zspec > z_low) & (zspec < z_high))[0] if zspec is not None else []
        good_zphot = np.where((zphot > z_low) & (zphot < z_high))[0] if zphot is not None else []
        
        # Only concatenate if both are available
        good = np.unique(np.concatenate((good_zspec, good_zphot)))  # Combine and remove duplicates
    else:
        raise ValueError("Invalid value for z_choice or missing redshift data. Ensure zspec or zphot are available.")

    # Filter catalog data based on the condition
    catalog_id = catalog_id[good]
    ra_ref = ra_ref[good]
    dec_ref = dec_ref[good]
    zspec = zspec[good]
    zfast = zphot[good]
    lmass = lmass[good]

    inner_col = f"{filter_name}_inner"
    outer_col = f"{filter_name}_outer"
    ninety_col = f"{filter_name}_90"
    
    # Create the output catalog file
    with open('z_member_catalogue.cat', 'w') as f:
        header = f"# old_id ID xcen ycen RA DEC semimaj semimin a_hl b_hl a_hl_90 b_hl_90 angle {inner_col} {outer_col} {ninety_col} lmass\n"
        f.write(header)
        # Prepare the data for output
        phdata = np.array([F160W_inner, F160W_outer, F160W_90])

        # Loop over the data and find the closest catalog source using srcor
        for i in range(len(ra_ref)):
            #print(i)
            # Use srcor to find the closest catalog entry to the current source (RA[i], DEC[i])
            ind1, ind2 = srcor(ra_ref[i], dec_ref[i], RA, DEC, dcr=dcr)

            # If no match is found, continue to the next iteration
            if len(ind1) == 0:
                continue

            # Compute angular distances for all matches
            if len(ind1) > 1:
                dist = np.sqrt((RA[ind2] - ra_ref[i])**2 + (DEC[ind2] - dec_ref[i])**2)
                closest_index = ind2[np.argmin(dist)]  # Choose the closest match

                f.write(f"{id1[ind2]} {catalog_id[i]} {xcenter[closest_index]:.4f} {ycenter[closest_index]:.4f} "
                        f"{RA[closest_index]:.6f} {DEC[closest_index]:.6f} {semimaj[closest_index]:.4f} {semimin[closest_index]:.4f} {a_hl[closest_index]:.4f} "
                        f"{b_hl[closest_index]:.4f} {a_hl_90[closest_index]:.4f} {b_hl_90[closest_index]:.4f} {angle[closest_index]:.4f} "
                        f"{phdata[0, closest_index]:.4f} {phdata[1, closest_index]:.4f} {phdata[2, closest_index]:.4f} {lmass[i]:.4f}\n")

            if len(ind1) == 1:
                f.write(f"{id1[ind2]} {catalog_id[i]} {xcenter[ind2]:.4f} {ycenter[ind2]:.4f} "
                        f"{RA[ind2]:.6f} {DEC[ind2]:.6f} {semimaj[ind2]:.4f} {semimin[ind2]:.4f} {a_hl[ind2]:.4f} "
                        f"{b_hl[ind2]:.4f} {a_hl_90[ind2]:.4f} {b_hl_90[ind2]:.4f} {angle[ind2]:.4f} "
                        f"{phdata[0, ind2]:.4f} {phdata[1, ind2]:.4f} {phdata[2, ind2]:.4f} {lmass[i]:.4f}\n")

    print("Output written to z_member_catalogue.cat")

def GalaxyAper_multi_filter(data, filter_data_map, catalog_matching_output=None, filter_name=FILTER):
    """
    Processes aperture photometry for multiple filters and saves the results into a CSV file. For each filter, it
    computes aperture photometry on the galaxy positions from a catalog and stores the photometric measurements for
    inner, outer, and r90 apertures.

    Parameters:
    -----------
    data : str or numpy.ndarray
        The input FITS data for a single filter. If a string is provided, it is assumed to be a file path to a FITS file.
        If a numpy array is provided, it represents the data array for the filter.
    
    filter_data_map : dict
        A dictionary where keys are filter names (e.g., 'F625W') and values are the corresponding FITS data arrays for
        each filter.
    
    catalog_matching_output : str, optional
        The path to the catalog file containing source positions. Defaults to 'z_member_catalogue.cat'.
    
    filter_name : str, optional
        The name of the filter used for catalog matching, included in the column names for the output. Defaults to `FILTER`.

    Returns:
    --------
    None
        The function writes the aperture photometry results for all filters to a CSV file (`full_array_iterative_all_filters.csv`).

    Notes:
    ------
    - The function assumes that the catalog data is available and processed using a segmentation map (`segmentation_map.fits`).
    - Aperture photometry is performed using three different aperture sizes: inner, outer, and r90.
    - The results for all filters are compiled into a single CSV file with columns for the photometric measurements across all filters.
    """
    if catalog_matching_output is None:
        catalog_matching_output = 'z_member_catalogue.cat'
    
    print(f"Using catalog_matching file: {catalog_matching_output}")
    table = ascii.read(catalog_matching_output)
    num_filters = len(filter_data_map)
    num_columns_per_filter = 3  # inner, outer, 90
    base_columns = 16  # ID, position, etc.

    data_csv = np.zeros((len(table), base_columns + num_filters * num_columns_per_filter))

    inner_col = f"{filter_name}_inner"
    outer_col = f"{filter_name}_outer"
    ninety_col = f"{filter_name}_90"
    
    # Fill the base columns
    for i in range(len(table)):
        data_csv[i, :16] = [
            table['ID'][i], table['xcen'][i], table['ycen'][i], table['RA'][i], table['DEC'][i],
            table['semimaj'][i], table['semimin'][i], table['a_hl'][i], table['b_hl'][i],
            table['a_hl_90'][i], table['b_hl_90'][i], table['angle'][i], table['lmass'][i],
            table[inner_col][i], table[outer_col][i], table[ninety_col][i]
        ]

    segm_deblend = fits.open('segmentation_map.fits')
    
    # Process each filter
    filter_start_index = base_columns
    for filter_name, data in filter_data_map.items():
        print(f"Processing filter: {filter_name}")
        for i in range(len(table)):
            mask = np.zeros(data.shape, dtype=bool)
            bad = np.where(segm_deblend[0].data != table['old_id'][i])
            ok = np.where(segm_deblend[0].data == 0)
            mask[bad] = True
            mask[ok] = False

            xypos = (table['xcen'][i], table['ycen'][i])
            theta = np.deg2rad(table['angle'][i])

            # Perform aperture photometry for inner, outer, and 90 apertures
            apertures = [
                EllipticalAperture(xypos, a=table['a_hl'][i], b=table['b_hl'][i], theta=theta),
                EllipticalAperture(xypos, a=table['semimaj'][i], b=table['semimin'][i], theta=theta),
                EllipticalAperture(xypos, a=table['a_hl_90'][i], b=table['b_hl_90'][i], theta=theta)
            ]
            
            for j, aper in enumerate(apertures):
                phot_table = aperture_photometry(data, aper, method='subpixel', mask=mask, subpixels=5)
                data_csv[i, filter_start_index + j] = phot_table['aperture_sum'][0].item()

        filter_start_index += num_columns_per_filter

    # Generate dynamic header
    header = ['id', 'xcenter', 'ycenter', 'xworld', 'yworld', 'semimajor', 'semiminor', 'a_hl',
              'b_hl', 'a_hl_90', 'b_hl_90', 'angle', 'lmass', inner_col, outer_col, ninety_col]
    for filter_name in filter_data_map.keys():
        header.extend([f"{filter_name}_inner", f"{filter_name}_outer", f"{filter_name}_90"])

    # Save the results
    output_file = 'full_array_iterative_all_filters.csv'
    with open(output_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_csv)






