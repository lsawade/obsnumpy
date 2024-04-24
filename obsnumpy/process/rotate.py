""""THIS IS A REMNANT OF THE OLD ROTATE FUNCTION. IT IS NOT USED AT THE MOMENT"""


def to_rtz(data, event_latitude, event_longitude):

    # Check if back_azimuth is set

    if (self.latitudes is None or self.longitudes is None:
        raise ValueError("Latitude and longitude must be set to convert to RTZ")

    # Compute azimuth and distance
    dist, az, baz = utils.distazbaz(
        event_latitude, event_longitude, self.latitudes, self.longitudes
    )

    print("D:", dist)
    print("AZ:", az)
    print("BAZ:", baz)

    # Convert azimuth to radians
    baz = np.radians(baz)

    # Compute rotation matrix
    # From obspy:
    #   r = - n * cos(ba) - e * sin(ba)
    #   t = + n * sin(ba) - e * cos(ba)
    R = np.array(
        [
            [-np.cos(baz), -np.sin(baz), np.zeros_like(baz)],
            [np.sin(baz), -np.cos(baz), np.zeros_like(baz)],
            [np.zeros_like(baz), np.zeros_like(baz), np.ones_like(baz)],
        ]
    )

    # Check shapes
    print("B shape:", baz.shape)
    print("R shape:", R.shape)
    print("D shape:", self.data.shape)

    # Rotate data
    self.data = np.einsum("ijk,kjl->kil", R, self.data)

    # Update components
    self.components = ["R", "T", "Z"]