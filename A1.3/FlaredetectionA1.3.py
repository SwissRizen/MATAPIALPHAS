# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 10:30:25 2021

@author: nils
"""
#%%################################# Modules and packages
                                                                        #¦
import streamlit as st

import sqlite3 as sq

import astropy.units as u

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np

import sunpy.map as smap
from sunpy.physics.solar_rotation import mapsequence_solar_derotate
from sunpy.net import Fido, attrs as a

from datetime import datetime, timedelta, date

from skimage import io
from skimage.metrics import structural_similarity

#%%################################# Database
                                                                        #¦
conn = sq.connect('flares.db')#connect to the database flares
c = conn.cursor() #create a cursor object needed to interact with it

#%%################################# Flask webapp
                                                                        #¦
if not hasattr(st, 'already_started_server'):
    # Streamlit only loads once. The first time this program executes,
    # it will launch the flask app; after reloading the real streamlit 
    # application is launched
    st.already_started_server = True

    st.write('''
        Please reload the page if you see this.
    ''') # this text will only appear the very first time it is started

    from gevent.pywsgi import WSGIServer #import the server library
    from appflask import app #import the flask application
    
    http_server = WSGIServer(('', 8888), app)
    http_server.serve_forever() #host "app" on Port 8888

#%%################################# Streamlitapp + Flarefinder
                                                                        #¦
@st.cache(show_spinner=True,suppress_st_warning=True)
def Flarefinder(start_datetime, end_datetime):
    #%%################################# Data Acquisition
    instrument = a.Instrument('AIA')
    wave = a.Wavelength(13*u.nm, 14*u.nm)

    start_datetime_d = start_datetime + timedelta(seconds=15)
    end_datetime_d = end_datetime + timedelta(seconds=15)
    result = Fido.search(
        a.Time(start_datetime, start_datetime_d) | 
        a.Time(end_datetime, end_datetime_d) , instrument, wave
        )
    downloaded_files = Fido.fetch(result, 
        path="static/FITS/"
        )
    try:  
        maps = smap.Map(downloaded_files, sequence = True)
    except RuntimeError:
        st.error('''No data was found. Please use an earlier time. 
            It usually takes about 2-3 days for the data to be 
            available'''
            )
        return
    timediff = end_datetime_d - start_datetime_d
    if timediff.total_seconds() >= 4*3600:
        maps = mapsequence_solar_derotate(maps)
    

    #%%################################# Dimensions to int
    dimstring = str(maps[-1].dimensions[0])
    dimint = int(float(dimstring.strip(' pix')))
    #%%################################# Superpixel
    pixamt = dimint/16
    newdim1= u.Quantity(maps[0].dimensions)/pixamt
    newdim2 = u.Quantity(maps[-1].dimensions)/pixamt
    spmap1 = maps[0].superpixel(newdim1)
    spmap2 = maps[-1].superpixel(newdim2)

    #%%################################# Difference image
                                                                        #¦
    diff = spmap1.data - spmap2.data
    metadiff= spmap2.meta
    diffmap= smap.Map(diff,metadiff)
    vdef = diffmap.max()*0.6
    fig = plt.figure()
    ax_diffmap= plt.subplot(projection = diffmap)
    dplot = diffmap.plot(cmap='Greys_r', 
        norm=colors.Normalize(vmin=-vdef, vmax=vdef))
    st.subheader("Difference Image:")
    st.write(fig)

    #%%################################# Flare Detection 
                                                                         #¦
    bar = diffmap.max()*0.98

    pixelpos = np.argwhere(abs(diffmap.data) >= bar)*u.pixel
    
    print('Possible flare locations:')
    print(pixelpos)
    print('Keep in mind here in pixel format it is y,x and not x,y')
        
    pixelcord = diffmap.pixel_to_world(pixelpos[:,1], pixelpos[:,0])
    print(pixelcord)

    #%%################################# Submap
                                                                         #¦
    pixoperator = (dimint/8, dimint/4)*u.pixel #set the subtrahend
    pixelpos4k = pixelpos * (dimint/pixamt) #revert the resolution 
    submapsize = 2 * maps[-1].scale[-1] * pixoperator #setting size of
                                                    # the result image
    
    st.subheader("Submaps:") #write subheader "Submaps"
    pics = [] #setting up a list called pics
    x = 0 #defining x as 0
    
    for flare in pixelpos: #for loop to display each flare
        now = datetime.utcnow()#get current time in UTC
        nowstr = now.strftime("%Y%b%athe%d%H%M%S") 
        startstr = start_datetime.strftime("%Y%b%athe%d%H%M%S")#string 
        endstr = end_datetime.strftime("%Y%b%athe%d%H%M%S")#conversions
        pixbot = pixelpos4k[x] - pixoperator #setting bottom left pixel
        cordbot = maps[-1].pixel_to_world(pixbot[1], pixbot[0]#convert
        submap = maps[-1].submap(cordbot, width= submapsize[1],
                                 height=submapsize[0])#create submap
        fig = plt.figure() #starting the figure
        ax_submap = plt.subplot(projection = submap)#axes
        submap.plot(cmap='sdoaia131')#plot with colormap SDOAIA131
        urlstr = ("static/Submaps/" + nowstr + "_" + startstr +
                "_" + endstr + ".jpeg") #create the url for image
        plt.savefig(urlstr) #save image

        #%%################################# SSIM
                                                                         #¦
        pics.append(urlstr) #append url to list pics
        if x == 0: #conditional if for first loop
            newflare = [startstr, endstr, urlstr] #create list newflare
            conn.execute('''insert into flares(stime, etime, urlfor) 
                        values (?,?,?) ''', newflare)#add flare to db
            conn.commit() #commit changes
            st.write(fig) #display image
            
        else: #if it is not the first loop
            pic1 = io.imread(pics[x-1], as_gray=True) #read previous pic 
            pic2 = io.imread(pics[x], as_gray=True) #read current pic
            ssim = structural_similarity(pic1,pic2) #compare the pics
            if ssim < 0.75: #if similarity is less than 0.75(max 1.0)
                newflare = [startstr, endstr, urlstr] #create list
                conn.execute('''insert into flares(stime, etime, urlfor) 
                            values (?,?,?) ''', newflare)#add flare to db
                conn.commit() #commit changes to db
                st.write(fig) #display
            else:
                print("Submap " + str(x+1) + "was not displayed(SSIM =" 
                    + str(ssim) + ")" ) #print text if ssim too big
   

#%%##################################################################### Streamlit app

st.title("Flare Detection WebApp")
st.write('''
Welcome to the Flare Detection Webapp. 
Instructions are simple, just put your desired date and time into the sidebars and press go. 
Please keep in mind that times should be in UTC and it might take a few days to update the database. 
''')
today = date.today()
tmrw = today + timedelta(days=1)

start_date = st.sidebar.date_input('Start date')
start_time = st.sidebar.time_input('Start time')
end_date = st.sidebar.date_input('End date')
end_time = st.sidebar.time_input('End time')

start_datetime = datetime.combine(start_date,start_time)
end_datetime = datetime.combine(end_date,end_time)

if start_datetime < end_datetime:
    st.success('Ready to go!')
else:
    st.error('Error: End Datetime must be later than Start Datetime')
    


start_datetime = datetime.combine(start_date,start_time)
end_datetime = datetime.combine(end_date,end_time)

st.write('Start Datetime: `%s`\n\nEnd Datetime:`%s`' % (start_datetime, end_datetime))


if st.sidebar.button('GO'):
    st.write('Running...')
    Flarefinder(start_datetime, end_datetime)














