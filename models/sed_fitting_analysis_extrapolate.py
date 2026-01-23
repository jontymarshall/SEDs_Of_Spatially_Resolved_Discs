#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:16:06 2025

@author: jonty
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from astropy.io import ascii
from RT_Code import RTModel
from scipy import interpolate

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 16}

plt.rc('font', **font)

direc = '/Users/jonty/mydata/robin/revised/extrapolate/'

filelist = np.asarray(os.listdir(direc+'../../json_files/'))
filelist = filelist[np.where(filelist != '.DS_Store')]

best_fit = ascii.read(direc + '../' + 'emcee_resolved_disc_tablulated_values_for_modelling.csv',delimiter=',')

#constants
h = 6.626e-34
c = 299792458.0 # m/s
k = 1.38e-23
sb = 5.67e-8 # 
au     = 1.495978707e11 # m 
pc     = 3.0857e16 # m
lsol   = 3.828e26 # W
rsol   = 6.96342e8 # m
MEarth = 5.97237e24 # kg

um = 1e-6 #for wavelengths in microns

def model(theta, x, y, yerr,radius, width):

    if len(theta) == 4:
        mdust, q, amin, epsilon = theta
    if len(theta) == 3:
        mdust, amin, epsilon = theta
        q = -3.5   

    model = RTModel()
    
    RTModel.get_parameters(model,'RTModel_Input_File.txt')

    model.parameters['directory'] = '/Users/jonty/mydata/robin/revised/extrapolate/'
    model.parameters['prefix'] = 'sed_extrapolating_'
    model.parameters['stype'] = 'json'
    model.parameters['model'] = direc + '../../json_files/' + target + '.json'
    model.parameters['tstar'] = sdbjson['main_results'][0]['Teff']
    model.parameters['rstar'] = sdbjson['main_results'][0]['rstar']
    model.parameters['dstar'] = (1/sdbjson['main_results'][0]['plx_arcsec'])
    model.parameters['lstar'] = sdbjson['main_results'][0]['lstar']
    
    model.obs_wave = x
    model.obs_flux = y  # [x*1000 for x in y]
    model.obs_uncs = yerr # [x*1000 for x in yerr]

    #Dust'
    model.parameters['composition'] = 'astrosil'
    model.parameters['density'] = 3.3
    model.parameters['dtype'] = 'gauss'
    model.parameters['mdust'] = mdust
    model.parameters['rpeak'] = radius
    model.parameters['rfwhm'] = width
    if len(theta) == 4:
        model.parameters['q'] = -1*q
    if len(theta) == 3:
        model.parameters['q'] = -3.5
    model.parameters['amin']  = amin
    
    RTModel.make_star(model)
    #RTModel.scale_star(model)
    RTModel.make_dust(model)
    RTModel.make_disc(model)
    RTModel.calculate_surface_density(model)
    RTModel.read_optical_constants(model)
    RTModel.calculate_qabs(model)
    RTModel.calculate_dust_emission(model,mode='full',tolerance=1e-2)
    RTModel.calculate_dust_scatter(model)
    RTModel.flam_to_fnu(model)
    
    mdl = (model.sed_wave, model.sed_star, model.sed_star + model.sed_emit, model.sed_emit)

    return mdl


fband1 = []
fband1_ep = []
fband1_em = []

fband6 = []
fband6_ep = []
fband6_em = []

for i in range(len(best_fit['Name'].data)):
    target = best_fit['Name'][i]
    
    file = open(direc+"../../json_files/{:}".format(target + '.json'))
    sdbjson = json.load(file)

    print("Processing target "+target+" which is "+str(int(i+1))+" of "+str(len(best_fit['Name'].data))+" targets.")

    #stellar parameters
    dstar = 1./sdbjson["main_results"][0]["plx_arcsec"]
    lstar = sdbjson["main_results"][0]["lstar"] 

    #stellar photosphere
    obs_photosphere = np.array(sdbjson['star_spec']['fnujy'])
    obs_lambda = np.array(sdbjson['star_spec']['wavelength'])

    #omit upper limits and duplicates from observations to be fitted
    ignore = np.asarray(sdbjson['phot_ignore'][0])
    upperlimit = np.asarray(sdbjson['phot_upperlim'][0])

    good = np.where((ignore == False)&(upperlimit == False))

    x = np.asarray(sdbjson['phot_wavelength'][0])[good]
    y = np.asarray(sdbjson['phot_fnujy'][0])[good]*1e3 #modelling in mJy, inputs in Jy
    yerr = np.asarray(sdbjson['phot_e_fnujy'][0])[good]*1e3 #modelling in mJy, inputs in Jy
    
    radius = best_fit['Radius'][i]
    width  = best_fit['Width'][i]
    
    #mdust, q, amin, epsilon = theta
    # theta = (best_fit['Md'][i],best_fit['q'][i],best_fit['smin'][i],1e-6)
    # output = model(theta, x, y, yerr,radius, width)
    # waves,fstar,ftotal,fdisc = output
    
    theta = (best_fit['Md'][i],best_fit['q'][i],best_fit['smin'][i],1e-6)
    output = model(theta, x, y, yerr,radius, width)
    waves,fstar,ftotal_mid,fdisc_mid = output
    
    f = interpolate.interp1d(waves,ftotal_mid)
    fb6 = f(1350.)
    fb1 = f(7000.)
    
    theta = (best_fit['Md'][i] - best_fit['Md_em'][i],best_fit['q'][i],best_fit['smin'][i],1e-6)
    output = model(theta, x, y, yerr,radius, width)
    waves,fstar,ftotal_low,fdisc_low = output
    
    f = interpolate.interp1d(waves,ftotal_low)
    fb6_lo = f(1350.)
    fb1_lo = f(7000.)
    
    theta = (best_fit['Md'][i] + best_fit['Md_ep'][i],best_fit['q'][i],best_fit['smin'][i],1e-6)
    output = model(theta, x, y, yerr,radius, width)
    waves,fstar,ftotal_hi,fdisc_hi= output
    
    f = interpolate.interp1d(waves,ftotal_hi)
    fb6_hi = f(1350.)
    fb1_hi = f(7000.)
    
    #fb6 in mJy, fb1 in uJy
    fband1.append(fb1)
    fband1_ep.append(fb1_hi - fb1)
    fband1_em.append(fb1 - fb1_lo)
    
    fband6.append(fb6)
    fband6_ep.append(fb6_hi - fb6)
    fband6_em.append(fb6 - fb6_lo)
    
    print(target,fb6,fb6_hi - fb6,fb6 - fb6_lo,fb1,fb1_hi - fb1,fb1 - fb1_lo)
    
    fig=plt.figure(figsize=(8,7))
    ax=fig.add_subplot(111)
    ax.errorbar(x,y,yerr=yerr,linestyle='',marker='o',color='k')
    ax.plot(waves,fstar,linestyle='--',marker='.',color='gray')
    ax.plot(waves,ftotal_hi,linestyle='-',marker='.',color='goldenrod')
    ax.plot(waves,ftotal_mid,linestyle='-',marker='.',color='green')
    ax.plot(waves,ftotal_low,linestyle='-',marker='.',color='blue')
    ax.plot([1350],[fb6],marker='o',color='green')
    ax.plot([7000],[fb1],marker='o',color='blue')
    ax.annotate(target,(1e3,1e3))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([0.1,10000.0])
    ax.set_ylim([0.0001,10000.0])
    ax.set_xlabel(r'Wavelength ($\mu$m)')
    ax.set_ylabel(r'Flux Density (mJy)')
    plt.show()
    

names_string = []
f1mm_string  = []
f10mm_string = []

for i in range(len(best_fit['Name'].data)):
    target = best_fit['Name'][i]
    names_string.append(target + ' & ')
    stra = "{0:#.3f}".format(fband6[i])
    strb = "{0:#.3f}".format(fband6_ep[i])
    strc = "{0:#.3f}".format(fband6_em[i])
    f1mm_string.append('$' + stra + '^{' + strb + '}_{' + strc + '}$ & ')
    stra = "{0:#.1f}".format(fband1[i]*1e3)
    strb = "{0:#.1f}".format(fband1_ep[i]*1e3)
    strc = "{0:#.1f}".format(fband1_em[i]*1e3)
    f10mm_string.append('$' + stra + '^{' + strb + '}_{' + strc + '}$ \\\\')

ascii.write([names_string,f1mm_string,f10mm_string],\
             direc+'emcee_resolved_disc_sed_flux_predictions.dat',\
             names=['Name', 'FBand6', 'FBand1'],delimiter=',',\
             overwrite=True)

ascii.write([names_string,fband6, fband6_ep, fband6_em],\
             direc+'band6_flux_density_predictions.dat',\
             names=['Name', 'F1.3mm','e+F1.3mm','e-F1.3mm'],delimiter=',',\
             overwrite=True)

#get source positions
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord

Simbad.reset_votable_fields()
Simbad.remove_votable_fields('coordinates')
Simbad.add_votable_fields('ra(:;A;ICRS;J2000)', 'dec(:;D;ICRS;2000)')

ra  = []
dec = []

for i in range(0,len(best_fit['Name'].data)):
    
    target = best_fit['Name'][i]
    
    if target == 'HD216956C':
        target = 'Fomalhaut C'
    
    result_table = Simbad.query_object(target)
    coords = SkyCoord(ra=['{}h{}m{}s'.format(*ra.split(':')) for ra in result_table['RA___A_ICRS_J2000']], 
                  dec=['{}d{}m{}s'.format(*dec.split(':')) for dec in result_table['DEC___D_ICRS_2000']],
                  frame='icrs', equinox='J2000')
    
    ra.append(coords.ra.value[0])
    dec.append(coords.dec.value[0])

#pick out names of targets with bright enough fluxes to be observed by NOEMA
for i in range(0,len(best_fit['Name'].data)):
    
    target = best_fit['Name'][i]
    
    file = open(direc+"../../json_files/{:}".format(target + '.json'))
    sdbjson = json.load(file)
    
    dstar = 1./sdbjson["main_results"][0]["plx_arcsec"]
    
    if fband6[i] > 0.5 and dec[i] > -15.0 :
        if ra[i] <= 60. or ra[i] >= 240.:
            print(best_fit['Name'].data[i],best_fit['Radius'].data[i]/dstar,best_fit['Lstar'].data[i],fband6[i],ra[i],dec[i])