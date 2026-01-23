#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:00:06 2024

@author: jonty
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import json
import os
from astropy.io import ascii
import astropy.coordinates as coord
import astropy.units as u
from astroquery.simbad import Simbad

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 16}

plt.rc('font', **font)

direc = '/Users/jonty/mydata/robin/revised/'

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist if not (x.startswith('.'))]

filelist = mylistdir(direc+'../json_files/')

print(filelist)


#Generate values from pickle files

targets = []
q     = []
q_ep  = []
q_em  = []
md    = []
md_ep = []
md_em = []
ad    = []
ad_ep = []
ad_em = []
dstar = []
lstar = []
mstar = []

composition = 'temp2'

burnin = 1750

nfail = 0
for f in filelist:
    target = f.split('.')[0].strip()
        
    try:
        #print('Processing target ',target)

        sdb = open(direc+"../json_files/{:}.json".format(target))
        sdbjson = json.load(sdb)
        
        with open(direc+ 'output_'+composition+'/'+target+'_flatchain.json', 'r') as payload:
            sampler = json.load(payload)
        
        sampler = np.asarray(sampler)
        
        ndim = sampler.shape[1]

        if ndim == 4:
            labels=["mdust", "q", "amin", "epsilon"]
            for i in range(0,ndim-1): 
               

               
                if i == 0:
                    quant = np.quantile(10**sampler[burnin::,i], [0.16, 0.5, 0.84], axis=0)
                    md.append(quant[1])
                    md_ep.append(quant[2] - quant[1])
                    md_em.append(quant[1] - quant[0]) 

                if i == 1:
                    quant = np.quantile(sampler[burnin::,i], [0.16, 0.5, 0.84], axis=0)
                    q.append(quant[1])
                    q_ep.append(quant[2] - quant[1])
                    q_em.append(quant[1] - quant[0])
                   
                if i == 2:
                    quant = np.quantile(10**sampler[burnin::,i], [0.16, 0.5, 0.84], axis=0)
                    ad.append(quant[1])
                    ad_ep.append(quant[2] - quant[1])
                    ad_em.append(quant[1] - quant[0])
            
        if ndim == 3:
            labels=["mdust", "amin", "epsilon"]
            for i in range(0,ndim-1): 
               
                quant = np.quantile(10**sampler[burnin::,i], [0.16, 0.5, 0.84], axis=0)
               
                if i == 0:
                    md.append(quant[1])
                    md_ep.append(quant[2] - quant[1])
                    md_em.append(quant[1] - quant[0]) 
                   
                if i == 1:
                    ad.append(quant[1])
                    ad_ep.append(quant[2] - quant[1])
                    ad_em.append(quant[1] - quant[0])
            
            q.append(-3.5)
            q_ep.append(0.0)
            q_em.append(0.0)

        #stellar parameters
        dstar.append(1./sdbjson["main_results"][0]["plx_arcsec"])
        lstar.append(sdbjson["main_results"][0]["lstar"])
        mstar.append(sdbjson["main_results"][0]["lstar"]**(1./3.5))
        targets.append(target)
        
    
    except:
        print('Target ',target,' sampler file not found.')
        nfail += 1
        
targets = np.asarray(targets)
dstar = np.asarray(dstar)
lstar = np.asarray(lstar)
mstar = np.asarray(mstar)
q     = abs(np.asarray(q))
q_ep  = np.asarray(q_ep)
q_em  = np.asarray(q_em)
md    = np.asarray(md)
md_ep = np.asarray(md_ep)
md_em = np.asarray(md_em)
ad    = np.asarray(ad)
ad_ep = np.asarray(ad_ep)
ad_em = np.asarray(ad_em)

rq = np.where(q != 3.5)

#Load Cao et al. 2023's data table - want names, ages

direc_cao = '/Users/jonty/mydata/cao2023/'
caotbl = ascii.read(direc_cao+'Table1.csv',delimiter=',')

#Remove footnotes
cao_name = caotbl['Names']
for i in range(len(cao_name)):
    cao_name[i] = cao_name[i].strip('$^{grp}')

cao_age = []
cao_tau = []

for target in targets: #
    
    try: 
        cao_index = np.where(cao_name == target)
        
        cao_tau.append(float(caotbl['TotalFracLum'][cao_index][0]))
        cao_age.append(float(caotbl['Age'][cao_index][0]))
        #print(target, caotbl['Age'][cao_index][0])
        
    except:
        if target == 'GJ581':
            cao_age.append(2000)
        elif target == 'HD89452':
            cao_age.append(100)
        elif target == 'GJ3760':
            cao_age.append(1000)
        elif target == 'HD189002':
            cao_age.append(2035)
        elif target == 'TWA7':
            cao_age.append(10)
        elif target == 'HD216956C':
            cao_age.append(440)
        elif target == 'TYC93404371':
            cao_age.append(23)
        elif target == 'GJ14':
            cao_age.append(500)
        elif target == 'HD142315':
            cao_age.append(11)
        elif target == 'HD37594':
            cao_age.append(3)
        elif target == 'HD176894':
            cao_age.append(2120)
        elif target == 'HD89452':
            cao_age.append(100)
        else :
            print('No age found for ',target,' adding fake')
            cao_age.append(-99)

cao_age = np.asarray(cao_age)
cao_tau = np.asarray(cao_tau)

qbig = 2.8
qmed = 3.7
smm  = 3e-3
s1km = 1e3
smax = 2e5
mdisk = md * ((4-q)/(4-qbig)) * (s1km/smm)**(4-qmed) * (smax/s1km)**(4-qbig)
mdisk_em = (md - md_em) * ((4-q)/(4-qbig)) * (s1km/smm)**(4-qmed) * (smax/s1km)**(4-qbig)
mdisk_ep = (md + md_ep) * ((4-q)/(4-qbig)) * (s1km/smm)**(4-qmed) * (smax/s1km)**(4-qbig)

ll = np.where((ad - 1.5*ad_em) < 0.01)
ul = np.where((ad + 1.5*ad_ep) > 100.)

print("Lower limits to amin: ",targets[ll])
print("Upper limits to amin: ",targets[ul])

#Plots vs age
t = np.arange(1,10000)
t0 = 10.
m0 = 300.
mt = m0*(1./t)**0.5

fig=plt.figure(figsize=(8,7))
ax=fig.add_subplot(111)
ax.errorbar(cao_age,mdisk,xerr=None,yerr=[mdisk_em,mdisk_ep],linestyle='',marker='o',color='black',mec='white',mfc='black')
ax.plot(t,mt,marker='',linestyle='-',color='red')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1,10000.0])
ax.set_ylim([1e-2,1e4])
ax.set_xlabel(r'Stellar Age (Myr)')
ax.set_ylabel(r'Total Disc Mass ($M_{\oplus}$)')
plt.draw()
plt.show()
fig.savefig(direc+'mdisk_vs_age_'+composition+'.pdf', dpi=200)
plt.close()

fig=plt.figure(figsize=(8,7))
ax=fig.add_subplot(111)
ax.errorbar(cao_tau,mdisk,xerr=None,yerr=[mdisk_em,mdisk_ep],linestyle='',marker='o',color='black',mec='white',mfc='black')
ax.plot(t,mt,marker='',linestyle='-',color='red')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1e-7,1e-2])
ax.set_ylim([1e-2,1e4])
ax.set_xlabel(r'Disc Fractional Luminosity ($L_{\rm d}/L_{\star}$)')
ax.set_ylabel(r'Total Disc Mass ($M_{\oplus}$)')
plt.draw()
plt.show()
fig.savefig(direc+'mdisk_vs_fraclum_'+composition+'.pdf', dpi=200)
plt.close()

fig=plt.figure(figsize=(8,7))
ax=fig.add_subplot(111)
ax.errorbar(cao_age[rq],q[rq],xerr=None,yerr=[q_em[rq],q_ep[rq]],linestyle='',marker='o',color='black',mec='white',mfc='black')
ax.set_xscale('log')
ax.set_yscale('linear')
ax.set_xlim([1,10000.0])
ax.set_ylim([5,2])
ax.set_xlabel(r'Stellar Age (Myr)')
ax.set_ylabel(r'Exponent of size distribution (-)')
plt.draw()
plt.show()
fig.savefig(direc+'q_vs_age_'+composition+'.pdf', dpi=200)
plt.close()


fig=plt.figure(figsize=(8,7))
ax=fig.add_subplot(111)
ax.errorbar(cao_age,ad,xerr=None,yerr=[ad_em,ad_ep],linestyle='',marker='o',color='black',mec='white',mfc='black')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1,10000.0])
ax.set_ylim([1e-2,1e2])
ax.set_xlabel(r'Stellar Age (Myr)')
ax.set_ylabel(r'Minimum grain size $a_{\rm min}$ ($\mu$m)')
plt.draw()
plt.show()
fig.savefig(direc+'mdisk_vs_age_'+composition+'.pdf', dpi=200)
plt.close()
