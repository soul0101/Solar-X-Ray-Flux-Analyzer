
import numpy as np
import pandas as pd
import streamlit as st
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy import timeseries as ts
from scipy.signal import savgol_filter
from sunpy.timeseries import TimeSeries

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime, timedelta

import sys, asyncio

if sys.platform == "win32" and (3, 8, 0) <= sys.version_info < (3, 9, 0):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
@st.cache()
def fetch_data(tstart, tend, columns=["xrsa", "xrsb"]):
    """
    Fetch the raw GOES X-Ray Sensor data.

    Parameters
    ----------
    tstart: datetime.datetime
    tend:   datetime.datetime
    columns: list
        List containing the spectral range for the raw data. 
        xrsa = 0.5-4 Å , xrsb = 1-8 Å

    Returns
    -------
    pandas.DataFrame: 
    """ 

    result = Fido.search(a.Time(tstart, tend), a.Instrument("XRS"))
    files = Fido.fetch(result)
    goes = ts.TimeSeries(files, concatenate=True)

    return goes.to_dataframe()[columns]

def noise_reduction_series(series, window_length=50):
    if type(series) == np.array:
        filtered_series = pd.Series(savgol_filter(series, window_length, 2), index=np.arange(len(a)))
    elif type(series) == pd.Series:
        filtered_series = pd.Series(savgol_filter(series, window_length, 2), index=series.index)
    else:
        raise TypeError("Function noise_reduction_series expects pd.Series or np.ndarray but received: %s" % (type(series)))

    return filtered_series

def noise_reduction_df(df, window_length=50):
    if type(df) != pd.DataFrame:
        raise TypeError("Function noise_reduction_df expects pd.Dataframe but received: %s" % (type(df)))
    for col in df:
        df[col] = noise_reduction_series(df[col], window_length=window_length)
    return df

def noise_reduction(data, window_length = 50):
    """
    Apply Savitzky-Golay filter to remove noise. 

    Parameters
    ----------
    data: pandas.DataFrame or pandas.Series or numpy.ndarray

    Returns
    -------
    pandas.DataFrame or pandas.Series or numpy.ndarray
        The filtered data

    """   
    if (type(data) == pd.Series or type(data) == np.array):
        return noise_reduction_series(data, window_length=window_length)
    elif type(data) == pd.DataFrame:
        return noise_reduction_df(data, window_length=window_length)
    else:
        raise TypeError("Function noise_reduction expects pd.Series or np.ndarray but received: %s" % (type(data)))
    
def get_flares(tstart, tend):
    """
    Fetch the flare times for given start and end datetime object

    Parameters
    ----------
    tstart: datetime.datetime
    tend:   datetime.datetime

    Returns
    -------
    pandas.DataFrame: 
        Dataframe containing the information for the flares in the given period. 
        [event_starttime (datetime.datetime) | event_peaktime (datetime.datetime) | event_endtime (datetime.datetime) | fl_goescls (str)]

    """   
    tend += timedelta(days=1)
    event_type = "FL"
    result = Fido.search(a.Time(tstart, tend),
                     a.hek.EventType(event_type),
                     a.hek.OBS.Observatory == "GOES")

    hek_results = result["hek"]["event_starttime", "event_peaktime", "event_endtime", "fl_goescls"]
    hek_results["event_starttime"] = hek_results["event_starttime"].tt.datetime
    hek_results["event_endtime"] = hek_results["event_endtime"].tt.datetime
    hek_results["event_peaktime"] = hek_results["event_peaktime"].tt.datetime
    return hek_results

def get_datetime_obj(date_str, date_format="%Y-%m-%d %H:%M"):
    """
    Convert date-string to datetime object

    Parameters
    ----------
    date_str: str
        format - "%Y-%m-%d %H:%M"
    Returns
    -------
    datetime.datetime
    """   
    return datetime.strptime(date_str, date_format)

def df_visualizer(flux_df) -> matplotlib.figure.Figure: 
    """
    Plots dataframe

    Parameters
    ----------
    flux_df: pandas.DataFrame
        Dataframe containing the flux rate with respect to time. 
    Returns
    -------
    matplotlib.figure.Figure
    """
    
    fig, ax = plt.subplots()
    plot_settings = {"xrsa": ["blue", r"0.5--4.0 $\AA$"], "xrsb": ["red", r"1.0--8.0 $\AA$"]}

    for col in flux_df.columns:
        ax.plot_date(
                flux_df.index, flux_df[col], "-", label=plot_settings[col][1], color=plot_settings[col][0], lw=2
            )
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y %H:00'))
    ax.set_yscale('log')
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right', fontsize='small')
    ax.set_ylabel('Watts m^-2')
    ax.legend()
    ax.set_ylim(1e-9, 1e-2)

    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(1e-9, 1e-2)
    labels = ["A", "B", "C", "M", "X"]
    centers = np.logspace(-7.5, -3.5, len(labels))
    ax2.yaxis.set_minor_locator(mticker.FixedLocator(centers))
    ax2.set_yticklabels(labels, minor=True)
    ax2.set_yticklabels([])
    ax.yaxis.grid(True, "major")
    ax.xaxis.grid(False, "major")
    return fig, ax

def flare_visualizer(flare_times, fig_flares, ax_flares) -> matplotlib.figure.Figure: 
    """
    Plots flares 

    Parameters
    ----------
    flare_times: pandas.DataFrame
        Dataframe containing the information for the flares in the given period. 
        event_starttime (datetime.datetime) | event_peaktime (datetime.datetime) | event_endtime (datetime.datetime) | fl_goescls (str)
    fig_flares: matplotlib.figure
    ax_flares: matplotlib.Axes

    Returns
    -------
    matplotlib.figure.Figure
    """

    ax_flares.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y %H:00'))
    for label in ax_flares.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    for flares_hek in flare_times:
        # ax_flares.axvspan(parse_time(flares_hek['event_starttime']).datetime,
        #             parse_time(flares_hek['event_endtime']).datetime,
        #             alpha=0.2, label=flares_hek['fl_goescls'])
        ax_flares.axvspan(flares_hek['event_starttime'],
                    flares_hek['event_endtime'],
                    alpha=0.2, label=flares_hek['fl_goescls'])

    return fig_flares, ax_flares

def st_ui():
    st.write("# Welcome to the GOES X-Ray Remote Sensing Daisi! 👋")
    st.markdown(
        """
        Cosmic sources in the sky, including our star, the Sun, burst intermittently in the X-ray energy
        band of the electromagnetic spectrum. The burst amplitude and the duration vary depending on
        the source and the cause of the burst. The observed burst has a fast rise and slow decay
        pattern/shape with variable rise and decay times. An automatic identification system for such
        data will further simplify the analysis process and the associated scientific investigation of
        these bursts.
        """
    )

    col1, col2 = st.columns(2)
    
    with col1:
        tstart = st.date_input(
            "Start Date",
            get_datetime_obj("2022-02-15 01:00"))
    with col2:
        tend = st.date_input(
            "End Date",
            get_datetime_obj("2022-02-15 01:00"))

    st.sidebar.header("Spectral Range")
    spct_rng = st.sidebar.radio("Spectral Range", ('1-8 Å', '0.5-4 Å', 'Both'), label_visibility="collapsed")
    
    if spct_rng == "1-8 Å":
        range_columns = ["xrsb"]
    elif spct_rng == "0.5-4 Å":
        range_columns = ["xrsa"]
    else:
        range_columns = ["xrsa", "xrsb"]

    analyze_btn = st.button("Analyze")

    if analyze_btn:
        with st.spinner("Downloading data..."):
            df = fetch_data(tstart, tend, columns=range_columns)[::10]
            
        fig_raw, ax_raw = df_visualizer(df)
        ax_raw.set_title('GOES X-Ray Flux (Raw)')
        st.header("Raw Data Input")
        st.write("The light curve is fetched and the ‘Time’ and ‘Rate’ values are plotted. ")
        st.pyplot(fig_raw)

        st.header("Data After Noise Reduction")
        st.write("Noise reduction is performed using the Savitzky-Golay filter. ")

        filtered_df = df.copy()
        filtered_df = noise_reduction(filtered_df)
        fig_filtered, ax_filtered = df_visualizer(filtered_df)
        ax_filtered.set_title('GOES X-Ray Flux (Filtered)')
        st.pyplot(fig_filtered)
        
        st.header("LC Solar Flare Classification")
        st.write("Solar flares are classified using the Soft X-ray classification or the H-alpha classification. The Soft X-ray classification is the modern classification system. Here 5 letters, A, B, C, M, or X are used according to the peak flux (Watt/m2). This classification system divides solar flares according to their strength. ")
        classification_df = pd.DataFrame({
            "Solar Flare Class": ["X", "M", "C", "B"],
            "Intensity W m⁻²": ["I > 10⁻⁴", "10⁻⁵ < I < 10⁻⁴", "10⁻⁶ < I < 10⁻⁵", "I < 10⁻⁶"]
        })

        st.markdown(classification_df.to_html(index=False), unsafe_allow_html=True)

        flare_times = get_flares(tstart, tend)

        fig_flares, ax_flares = df_visualizer(filtered_df)  
        fig_flares, ax_flares = flare_visualizer(flare_times, fig_flares, ax_flares)
        ax_flares.set_title('Solar Flares Classification')
        st.pyplot(fig_flares)
        st.table(pd.DataFrame(flare_times, columns=flare_times.columns))

if __name__ == '__main__':
    st_ui()