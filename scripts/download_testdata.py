import os
import obspy
from obsnumpy.download import download_waveforms_to_storage
import matplotlib.pyplot as plt

scriptfolder = os.path.dirname(os.path.abspath(__file__))
datafolder = os.path.join(scriptfolder, "data")

# Get event
eventfile = os.path.join(scriptfolder, "germany_event.cmt")
event = obspy.read_events(eventfile)[0]

# Data and station storage
if not os.path.exists(datafolder):
    os.makedirs(datafolder)

# Get bounding box (only stations within this box will be downloaded)
minlat = 35
maxlat = 55
minlon = -7.5
maxlon = 17.5

# Get event data
origin = event.origins[0]
event_latitude = origin.latitude
event_longitude = origin.longitude
starttime = origin.time - 100
endtime = starttime + 1100

# # Download
download_waveforms_to_storage(
    datafolder,
    starttime=starttime,
    endtime=endtime,
    minlatitude=minlat,
    maxlatitude=maxlat,
    minlongitude=minlon,
    maxlongitude=maxlon,
    channel_priorities=["BH[ZNE]", "HH[ZNE]", "EH[ZNE]"],
    minimum_length=0.95,
)

# Preprocess the data
inv = obspy.read_inventory(os.path.join(datafolder, "stations", "*.xml"))
st = obspy.read(os.path.join(datafolder, "waveforms", "*.mseed"))

# Basic preprocessing
st.detrend("linear")
st.detrend("demean")
st.taper(max_percentage=0.05, type="cosine")

# Set bandpass filter between 0.001 and 0.005 Hz and 45 and 50 Hz
pre_filt = [0.05, 0.1, 1, 2]

# Remove response
st.remove_response(
    inventory=inv, pre_filt=pre_filt, output="VEL", water_level=60, plot=True
)

plt.show(block=True)

# Move to NE
st.rotate(method="->ZNE", inventory=inv)

# Finally interpolate to maximum prefilt * 2.5
sampling_rate = pre_filt[3] * 2.5

# Store raw preprocessed
if not os.path.exists(os.path.join(datafolder, "preprocessed")):
    os.makedirs(os.path.join(datafolder, "preprocessed"))

# Store waveforms
for tr in st:
    tr.write(
        os.path.join(datafolder, "preprocessed", tr.id + ".mseed"),
        format="MSEED",
    )

# Store interpolated
st.interpolate(
    sampling_rate=sampling_rate, starttime=origin.time, npts=int(1000 * sampling_rate)
)

# Store interp
if not os.path.exists(os.path.join(datafolder, "interp")):
    os.makedirs(os.path.join(datafolder "interp"))

# Store waveforms
for tr in st:
    tr.write(
        os.path.join(datafolder, "interp", tr.id + ".mseed"),
        format="MSEED",
    )
