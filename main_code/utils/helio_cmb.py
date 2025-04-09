def perform_corr(z_inp, RA, Dec, corrtype="full", dipole="Planck"):

    """
    A function to perform heliocentric corrections on redshifts.
    Inputs:
        z: float, input heliocentric or CMB frame redshift
        RA: float, equatorial right ascension
        Dec: float, equatorial declination
        corrtype: string of the type of correction to be performed. Either 'full' (default), 'approx',
            '-full' or '-approx', respectively corresponding to the proper correction, low-z additive
            approximation, and the backwards corrections to go from z_CMB to z_helio.
        dipole: string specifying 'Planck', 'COBE' or 'Astropy' dipoles. 'Astropy' imports astropy (slow)
                and uses their co-ordinates and transforms and COBE dipole; different beyond first decimal
                point.
    Outputs:
        z_CMB (if corrtype is 'full' or 'approx'): float
        z_helio (only if corrtype is '-full' or '-approx'): float

    Notes:
      Co-ords of North Galactic Pole (ICRS): RA = 192.729 ± 0.035 deg, Dec = 27.084 ± 0.023 deg (https://doi.org/10.1093/mnras/stw2772)
      Co-ords of Galactic Centre (ICRS): RA = 17h45m40.0409s, Dec = −29d00m28.118s (see above reference)
                                         RA = 266.41683708 deg, Dec = -29.00781056 deg
      Ascending node of the galactic plane = arccos(sin(Dec_GC)*cos(Dec_NGP)-cos(Dec_GC)*sin(Dec_NGP)*cos(RA_NGP-RA_GC))
                                           = 122.92828126730255 = l_0
      Transform CMB dipole from (l,b) to (RA,Dec):
          Dec = arcsin(sin(Dec_NGP)*sin(b)+cos(Dec_NGP)*cos(b)*cos(l_0-l))
              = -6.9895105228347 deg
          RA = RA_NGP + arctan((cos(b)*sin(l_0-l)) / (cos(Dec_NGP)*sin(b)-sin(Dec_NGP)*cos(b)*cos(l_0-l)))
             = 167.81671014708002 deg
      Astropy co-ordinates:
      RA_NGP_J2000 = 192.8594812065348, Dec_NGP_J2000 = 27.12825118085622, which are converted from B1950
      RA_NGP_B1950 = 192.25, Dec_NGP_B1950 = 27.4
      l_0_B1950 = 123
      l_0_J2000 = 122.9319185680026
    """

    import numpy as np

    v_Sun_Planck = 369.82  # +/- 0.11 km/s
    l_dipole_Planck = 264.021  # +/- 0.011 deg
    b_dipole_Planck = 48.253  # +/- 0.005 deg
    c = 299792.458  # km/s
    v_Sun_COBE = 371.0
    l_dipole_COBE = 264.14
    b_dipole_COBE = 48.26

    RA_Sun_Planck = 167.81671014708002  # deg
    Dec_Sun_Planck = -6.9895105228347
    RA_Sun_COBE = 167.88112630619747  # deg
    Dec_Sun_COBE = -7.024553155965497

    # RA_Sun_COBE = 168.01187366045565  # deg using Astropy values     # 168.0118667
    # Dec_Sun_COBE = -6.983037861854297  # # # -6.98303424

    if corrtype not in ["full", "approx", "-full", "-approx"]:
        print("Correction type unknown.")
        raise ValueError

    rad = np.pi / 180.0
    if dipole == "Planck":
        # Vincenty formula
        alpha = np.arctan2(
            np.hypot(
                np.cos(Dec_Sun_Planck * rad) * np.sin(np.fabs(RA - RA_Sun_Planck) * rad),
                np.cos(Dec * rad) * np.sin(Dec_Sun_Planck * rad)
                - np.sin(Dec * rad)
                * np.cos(Dec_Sun_Planck * rad)
                * np.cos(np.fabs(RA - RA_Sun_Planck) * rad),
            ),
            np.sin(Dec * rad) * np.sin(Dec_Sun_Planck * rad)
            + np.cos(Dec * rad)
            * np.cos(Dec_Sun_Planck * rad)
            * np.cos(np.fabs((RA - RA_Sun_Planck)) * rad),
        )
    elif dipole == "COBE":
        alpha = np.arctan2(
            np.hypot(
                np.cos(Dec_Sun_COBE * rad) * np.sin(np.fabs(RA - RA_Sun_COBE) * rad),
                np.cos(Dec * rad) * np.sin(Dec_Sun_COBE * rad)
                - np.sin(Dec * rad)
                * np.cos(Dec_Sun_COBE * rad)
                * np.cos(np.fabs(RA - RA_Sun_COBE) * rad),
            ),
            np.sin(Dec * rad) * np.sin(Dec_Sun_COBE * rad)
            + np.cos(Dec * rad)
            * np.cos(Dec_Sun_COBE * rad)
            * np.cos(np.fabs((RA - RA_Sun_COBE)) * rad),
        )
    elif dipole == "astropy":
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        coord_helio_COBE = SkyCoord(
            l_dipole_COBE * u.degree, b_dipole_COBE * u.degree, frame="galactic"
        ).galactic
        coord = SkyCoord(RA * u.degree, Dec * u.degree, frame="icrs").galactic
        alpha = coord_helio_COBE.separation(coord).radian
    if dipole == "Planck":
        v_Sun_proj = v_Sun_Planck * np.cos(alpha)
    elif dipole == "COBE" or dipole == "astropy":
        v_Sun_proj = v_Sun_COBE * np.cos(alpha)

    # z_Sun = -v_Sun_proj / c
    # Full special rel. correction since it is a peculiar vel
    z_Sun = np.sqrt((1.0 + (-v_Sun_proj) / c) / (1.0 - (-v_Sun_proj) / c)) - 1.0

    min_z = 0.0
    if corrtype == "full":
        z_CMB = np.where(z_inp <= min_z, z_inp, (1 + z_inp) / (1 + z_Sun) - 1)
    elif (corrtype == "-full"):
            # backwards correction where z_CMB is actually z_helio and vice versa
        z_helio = np.where(z_inp <= min_z, z_inp, (1 + z_inp) * (1 + z_Sun) - 1)
    elif corrtype == "approx":
        z_CMB = np.where(z_inp <= min_z, z_inp, z_inp - z_Sun)
    elif corrtype == "-approx":
        z_helio = np.where(z_inp <= min_z, z_inp,  z_inp + z_Sun)

    if corrtype[0] == "-":
        return z_helio
    else:
        return z_CMB
