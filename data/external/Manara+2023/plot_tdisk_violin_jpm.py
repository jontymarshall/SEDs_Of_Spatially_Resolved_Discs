
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys,os,pdb
from lifelines import KaplanMeierFitter
from astropy.table import Table
import pandas as pd
from scipy.optimize import curve_fit
import scipy.integrate as integrate
from astropy.io import ascii

def KM_C(T,E):

    #E,T = np.loadtxt(fname=file, unpack=True) # E = 1 --> DET, E = 0 --> UL
    dfrac = float((E==1).sum())/E.size
    kmf = KaplanMeierFitter()
    kmf.fit(T, E, alpha=0.32)   # 68% confidence limits = 1 sigma

    k = kmf.survival_function_
    x = np.array(k.index)
    y = dfrac*np.array(k.values[:,0])
    
    kc = kmf.confidence_interval_
    yu = dfrac*kc.values[:,0]
    yl = dfrac*kc.values[:,1]

    return x,y,yu,yl


def KM_C_UL(T,E):
    # here I use the version of KMF that includes both upper an lower limits

    #print(file)

    #E,en,T = np.loadtxt(fname=file, unpack=True) # E = 1 --> DET, E = 0 --> UL (or lower limit if entry>0)
    dfrac = float((E==1).sum())/E.size
    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E, entry=np.zeros(len(T)), alpha=0.32)   # 68% confidence limits = 1 sigma

    k = kmf.survival_function_
    x = np.array(k.index)
    y = dfrac*np.array(k.values[:,0])
    
    kc = kmf.confidence_interval_
    yu = dfrac*kc.values[:,0]
    yl = dfrac*kc.values[:,1]

    return x,y,yu,yl


def gauss_cdf(x, mean, sigma):
    # integrate gaussian, normalized so sum(0-infinity) = 1
    const = np.sqrt(1/(2*np.pi))
    xnorm = (x-mean)/sigma
    I = np.array([integrate.quad(lambda z: const/np.exp(0.5*z**2), -9, x_)[0] for x_ in xnorm])
    return I


def gauss_fit(x, y, fit=False, ax=None):
    #ifit = np.isfinite(yflag)
    #positives = x > 0
    #mask = ifit & positives
    ifit = x > 0
    xfit = np.log10(x[ifit])
    yfit = 1-y[ifit]
    pfit, pcov = curve_fit(gauss_cdf, xfit, yfit)
    perr = np.sqrt(np.diag(pcov))
    if fit:
        try:
            # plot on the given axis
            ax.plot(10**xfit, 1-gauss_cdf(xfit, pfit[0], pfit[1]), 'w--', lw=2)
        except:
            # plot on the default axis
            plt.plot(10**xfit, 1-gauss_cdf(xfit, pfit[0], pfit[1]), 'w--', lw=2)
    return pfit, perr

def gauss_pdf(x, p):
    return np.exp(-(x-p[0])**2/(2*p[1]**2)) / (np.sqrt(2*np.pi)*p[1])

def get_range_pdf(x, p0, p1, p2):
    nx = x.size

    y0 = gauss_pdf(x, [p0[0], p0[1]])
    y1 = gauss_pdf(x, [p1[0], p1[1]])
    y2 = gauss_pdf(x, [p2[0], p2[1]])
    ymin = np.array([y0, y1, y2]).min(axis=0)
    ymax = np.array([y0, y1, y2]).max(axis=0)

    return ymin, ymax


# model with dust
direc = '/Users/jonty/mydata/robin/revised/ppd_masses/manara_plot/'
data_dust_macc = np.genfromtxt(direc+'dust_gamma1p0_0.001_0.1_30_smooth_mdot.dat') #np.genfromtxt('/Users/cmanara/work/papers/ppvii_chapter_disc_ev/models/dust_gamma1p0_0.001_0.1_30_smooth_mdot.dat')
data_mdust = np.genfromtxt(direc+'mass_dust.txt') #np.genfromtxt('/Users/cmanara/work/papers/ppvii_chapter_disc_ev/models/mass_dust.txt')

earth_to_sun = 332948.344 # conversion from Earth to Sun Masses


Age = [0.8,1.,1.5,2.,4.,7.]
Region = ['Ophiuchus', 'Taurus',  'Lupus','ChaI', 'ChaII', 'UpperSco']
color = ['red','orange',  'green', 'steelblue','cyan', 'darkviolet']

# Age = [1.5,1.,0.8,2.,7.,4.]
# Region = ['Lupus', 'Taurus', 'Ophiuchus', 'ChaI', 'UpperSco', 'ChaII']
# color = ['green', 'orange', 'red', 'steelblue', 'darkviolet','cyan']

manara23_data = ascii.read(direc+'../PP7-Surveys_2022-10-19_PPVII_website.csv',delimiter=',',comment='#',guess=False,data_start=1)

regions = manara23_data['Region'].data
regnames = np.unique(regions)
classes = manara23_data['Disk'].data
detect  = manara23_data['Disk_Flag'].data
masses  = manara23_data['Standardized_Mdust_Mearth'].data
surveys = manara23_data['Survey_Ref'].data

plotflag = True

# fig = plt.figure(figsize=(14,6))
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)

# for ax in [ax1,ax2]:
#     ax.set_xscale('log')
#     ax.set_xlim(0.01,200)
#     ax.set_xticks([0.1, 1, 10, 100])
#     ax.set_xticklabels(['0.1','1','10','100'])
#     ax.set_ylim(0.0,1.0)
#     ax.tick_params(which='minor',axis='y',length=3,color='k',width=1.5)
#     ax.minorticks_on()
# # ax2.set_ylim(0.0,0.6)

# ax1.set_ylabel(r'$\mathregular{P \geq t_{disk}}$',fontsize=17)
# ax1.set_xlabel(r'$\mathregular{t_{disk}}$'+' '+r'$[Myr]$',fontsize=17)
# ax2.set_ylabel(r'$\mathregular{P(M_{dust})}$',fontsize=17)
# ax2.set_xlabel(r'$\mathregular{t_{disk}}$'+' '+r'$\mathregular{[Myr]}$',fontsize=17)
# # ax.set_xlim(0.005,200)

# logmu  = np.ones(len(Region))
# logsig = np.ones(len(Region))
# lx_pdf = np.linspace(-2.5,3,100)
# x_pdf = 10**lx_pdf
# for ir,name1 in enumerate(Region):
#     c = color[ir]
    
#     reg = name1
#     if name1 == 'Ophiuchus':
#         reg = 'rOph'
#         obs = masses[np.where((regions == reg)&(classes == 'II'))]
#         det = detect[np.where((regions == reg)&(classes == 'II'))].astype('bool')
#     if name1 == 'ChaI':
#         reg = 'ChamI'
#         obs = masses[np.where((regions == reg)&(classes == 'II'))]
#         det = detect[np.where((regions == reg)&(classes == 'II'))].astype('bool')
#     if name1 == 'ChaII':
#         reg = 'ChamII'
#         obs = masses[np.where((regions == reg)&(classes != 'Debris'))]
#         det = detect[np.where((regions == reg)&(classes != 'Debris'))].astype('bool')
#     if name1 == 'UpperSco':
#         reg = 'USco'
#         obs = masses[np.where((regions == reg)&(classes != 'Debris'))]
#         det = detect[np.where((regions == reg)&(classes != 'Debris'))].astype('bool')
#     if name1 == 'ChaII':
#         obs = masses[np.where((regions == reg)&(classes == 'II'))]
#         det = detect[np.where((regions == reg)&(classes == 'II'))].astype('bool')
#         good = np.where(obs != '--')
#         obs = obs[good]
#         det = det[good]
#     if name1 == 'Taurus':
#         obs = masses[np.where((regions == reg)&(classes == 'II'))]
#         det = detect[np.where((regions == reg)&(classes == 'II'))].astype('bool')
    
#     for io in range(0,len(obs)):
#         if '<' in obs[io]:
#             obs[io] = float(obs[io].strip('<'))
#         else:
#             obs[io] = float(obs[io])
    
#     x, y, yl, yu = KM_C(obs,det)
#     good = (yl < yu) | (yl == 0)
#     x = x[good]
#     y = y[good]
#     yl = yl[good]
#     yu = yu[good]
#     ax1.fill_between(x, yl, yu, facecolor=c, alpha=0.5, lw=0, label=name1)
#     print('-'*40)
#     p, perr = gauss_fit(x, y, fit=plotflag, ax=ax1)
#     mu = 10**p[0]
#     siglog = p[1]
#     pl, perr = gauss_fit(x, yl, fit=False)
#     mu_lo = 10**pl[0]
#     siglog_lo = pl[1]
#     pu, perr = gauss_fit(x, yu, fit=False)
#     mu_hi = 10**pu[0]
#     siglog_hi = pu[1]
#     ymin, ymax = get_range_pdf(lx_pdf, p, pl, pu)
#     ax2.plot(x_pdf, gauss_pdf(lx_pdf, p), color=c, lw=3)
#     #ax2.fill_between(x_pdf, ymin, ymax, facecolor=c, alpha=0.5, lw=0)
#     logmu[ir] = np.log10(mu)
#     logsig[ir] = siglog
#     print(name1)
#     print('    meantnu  = {0:5.2f}  (+ {1:5.2f} - {2:5.2f})'.format(mu, mu_hi-mu, mu-mu_lo))
#     print('sigma_logtnu = {0:5.2f}  (+ {1:5.2f} - {2:5.2f})'.format(siglog, siglog_hi-siglog, siglog-siglog_lo))

# print('-'*40)

# l = ax1.legend(borderaxespad=1.2, fancybox=True, shadow=True, prop={'size': 12})
# l.set_title('Class II',prop={'size':14})

# plt.tight_layout()
# plt.savefig(direc+'PP7_tdisk.png', dpisize=600)
# plt.show()





# name = ['Lupus', 'Taurus', 'Ophiuchus', 'ChaI', 'UpperSco', 'ChaII']
# color = iter(['green', 'orange', 'red', 'steelblue', 'darkviolet','cyan'])
# logmu  = np.log10(np.array([1.07,0.22,0.02,0.22,0.42,1.35]))
# logsig = np.array([1.05,0.94,1.38,1.76,1.21,0.04])


# fig = plt.figure(figsize=(8,5))
# ax = fig.add_subplot(111)

# ax.set_xlabel('Region Age [Myr]', fontsize=18, labelpad=8)
# ax.set_ylabel(r'log $t_{\rm disk} = M_{\rm disk}/\dot{M}_{\rm acc}$ [Myr]', fontsize=18, labelpad=8)
# # ax.set_ylim([-3.8, 3])
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlim([0.5 , 25])
# ax.set_ylim([10**3 , 10**(-3)])

# lx_pdf, f = np.linspace(-8,4,60), 0.4*np.array(Age)
# for i, val in enumerate(Region):
#     if val == 'ChaII':
#         continue
#     y = gauss_pdf(lx_pdf, (logmu[i], logsig[i]))
#     ax.fill_betweenx(10**lx_pdf, Age[i]-f[i]*y, Age[i]+f[i]*y, facecolor=color[i], lw=0, zorder=2, alpha=0.5,label=val)
    
# # ax.text(0.6, -3.9, r'Oph')
# # ax.text(0.85, -3.9, r'Taurus')
# # ax.text(1.3, -3.9, r'Lupus')
# # ax.text(1.85, -3.9, r'Cham I')
# # ax.text(6.2, -3.9, r'USco')

# l = ax.legend(borderaxespad=1.2, fancybox=True, shadow=True, prop={'size': 12})
# l.set_title('Class II',prop={'size':14})
    


# x = np.linspace(0.001,20.)

# ### PLOT td=age
# plt.loglog(x,x,'-',color='orange',lw=3,zorder=15)


# ### PLOT A VISCOUS MODEL EXPECTATION
# tnu = 1. # Myr
# gamma = 1.7#9
# plt.loglog(x,2*(2-gamma)*(x+tnu),'k-.',lw=1.5,label=r'$t_{\nu}$ = %.1f, $\gamma$ = %.1f' % (tnu,gamma),zorder=10)

# tnu = 1. # Myr
# gamma = 1.5#9
# plt.loglog(x,2*(2-gamma)*(x+tnu),'b-.',lw=1.5,label=r'$t_{\nu}$ = %.1f, $\gamma$ = %.1f' % (tnu,gamma),zorder=10)

# # tnu = 1. # Myr
# # gamma = 1.#9
# # plt.loglog(x,2*(2-gamma)*(x+tnu),'r-.',lw=1.5,label=r'$t_{\nu}$ = %.1f, $\gamma$ = %.1f' % (tnu,gamma),zorder=10)

# # tnu = 0.5 # Myr
# # gamma = 1.7#9
# # plt.loglog(x,2*(2-gamma)*(x+tnu),'k--',lw=1.5,label=r'$t_{\nu}$ = %.1f, $\gamma$ = %.1f' % (tnu,gamma),zorder=10)

# # tnu = 0.5 # Myr
# # gamma = 1.5#9
# # plt.loglog(x,2*(2-gamma)*(x+tnu),'b--',lw=1.5,label=r'$t_{\nu}$ = %.1f, $\gamma$ = %.1f' % (tnu,gamma),zorder=10)

# # tnu = 0.5 # Myr
# # gamma = 1.#9
# # plt.loglog(x,2*(2-gamma)*(x+tnu),'r--',lw=1.5,label=r'$t_{\nu}$ = %.1f, $\gamma$ = %.1f' % (tnu,gamma),zorder=10)

# # tnu = 0.1 # Myr
# # gamma = 1.7#9
# # plt.loglog(x,2*(2-gamma)*(x+tnu),'k-',lw=1.5,label=r'$t_{\nu}$ = %.1f, $\gamma$ = %.1f' % (tnu,gamma),zorder=10)

# # tnu = 0.1 # Myr
# # gamma = 1.5#9
# # plt.loglog(x,2*(2-gamma)*(x+tnu),'b-',lw=1.5,label=r'$t_{\nu}$ = %.1f, $\gamma$ = %.1f' % (tnu,gamma),zorder=10)

# # tnu = 0.1 # Myr
# # gamma = 1.#9
# # plt.loglog(x,2*(2-gamma)*(x+tnu),'r-',lw=1.5,label=r'$t_{\nu}$ = %.1f, $\gamma$ = %.1f' % (tnu,gamma),zorder=10)

# # ### PLOT A MHD WIND MODEL EXPECTATION
# # # tacc = 0.3 # Myr
# # # fm = 0.6
# # # plt.loglog(x,np.repeat(2*tacc*(1+fm) , len(x)),'r--',lw=1.5)#,label=r'$t_{\rm acc}$ = %.1f, $f_M$ = %.1f' % (tacc,fm))

# fm = 1

# # tacc = 0.1 # Myr
# # plt.loglog(x,np.repeat(2*tacc*(1+fm) , len(x)),'g-.',lw=2)#,label=r'$t_{\rm acc}$ = %.1f, $f_M$ = %.1f' % (tacc,fm),zorder=10)

# tacc = 0.5 # Myr
# plt.loglog(x,np.repeat(2*tacc*(1+fm) , len(x)),'g--',lw=2)#,label=r'$t_{\rm acc}$ = %.1f, $f_M$ = %.1f' % (tacc,fm),zorder=10)

# tacc = 1 # Myr
# plt.loglog(x,np.repeat(2*tacc*(1+fm) , len(x)),'g-',lw=2)#,label=r'$t_{\rm acc}$ = %.1f, $f_M$ = %.1f' % (tacc,fm),zorder=10)


# # add the expectation from dust evolution
# # plt.loglog(np.linspace(1e5,3.e6,len(data_mdust))/1e6,100*data_mdust/earth_to_sun/data_dust_macc[:,1]/1e6, 'k--',lw=3,label='DUST')
# plt.loglog(data_dust_macc[:,0],100*data_mdust/earth_to_sun/data_dust_macc[:,1]/1e6, '--',color='orange',lw=3)#,label='DUST')



# fig.tight_layout()
# fig.savefig(direc+'PP7_tdisk_violin.pdf', bbox_inches='tight', dpi=200, alpha=True, rasterized=True)

# plt.show()


#########
#########
#########
### NOW CONSIDERING ALSO THE LOWER LIMITS
#########
#########
#########
Age = [0.8,1.,1.5,2.,4.,7.]
Region = ['Ophiuchus', 'Taurus',  'Lupus','ChaI', 'ChaII', 'UpperSco']
color = ['red','orange',  'green', 'steelblue','cyan', 'darkviolet']

fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for ax in [ax1,ax2]:
    ax.set_xscale('log')
    ax.set_xlim(0.01,200)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(['0.1','1','10','100'])
    ax.set_ylim(0.0,1.0)
    ax.tick_params(which='minor',axis='y',length=3,color='k',width=1.5)
    ax.minorticks_on()
# ax2.set_ylim(0.0,0.6)

ax1.set_ylabel(r'$\mathregular{P \geq t_{disk}}$',fontsize=17)
ax1.set_xlabel(r'$\mathregular{t_{disk}}$'+' '+r'$[Myr]$',fontsize=17)
ax2.set_ylabel(r'$\mathregular{P(M_{dust})}$',fontsize=17)
ax2.set_xlabel(r'$\mathregular{t_{disk}}$'+' '+r'$\mathregular{[Myr]}$',fontsize=17)
ax.set_xlim(0.005,200)

logmu_ul  = np.ones(len(Region))
logsig_ul = np.ones(len(Region))
lx_pdf = np.linspace(-2.5,3,100)
x_pdf = 10**lx_pdf
for ir,name1 in enumerate(Region):
    c = color[ir]
    
    reg = name1
    if name1 == 'Ophiuchus':
        reg = 'rOph'
        obs = masses[np.where((regions == reg)&(classes == 'II'))]
        det = detect[np.where((regions == reg)&(classes == 'II'))].astype('bool')
    if name1 == 'ChaI':
        reg = 'ChamI'
        obs = masses[np.where((regions == reg)&(classes == 'II'))]
        det = detect[np.where((regions == reg)&(classes == 'II'))].astype('bool')
    if name1 == 'ChaII':
        reg = 'ChamII'
        obs = masses[np.where((regions == reg)&(classes != 'Debris'))]
        det = detect[np.where((regions == reg)&(classes != 'Debris'))].astype('bool')
    if name1 == 'UpperSco':
        reg = 'USco'
        obs = masses[np.where((regions == reg)&(classes != 'Debris'))]
        det = detect[np.where((regions == reg)&(classes != 'Debris'))].astype('bool')
    if name1 == 'ChaII':
        obs = masses[np.where((regions == reg)&(classes == 'II'))]
        det = detect[np.where((regions == reg)&(classes == 'II'))].astype('bool')
        good = np.where(obs != '--')
        obs = obs[good]
        det = det[good]
    if name1 == 'Taurus':
        obs = masses[np.where((regions == reg)&(classes == 'II'))]
        det = detect[np.where((regions == reg)&(classes == 'II'))].astype('bool')
    
    for io in range(0,len(obs)):
        if '<' in obs[io]:
            obs[io] = float(obs[io].strip('<'))
        else:
            obs[io] = float(obs[io])
    
    x, y, yl, yu = KM_C_UL(obs,det)
    good = (yl < yu) | (yl == 0)
    x = x[good]
    y = y[good]
    yl = yl[good]
    yu = yu[good]
    ax1.fill_between(x, yl, yu, facecolor=c, alpha=0.5, lw=0, label=name1)
    print('-'*40)
    p, perr = gauss_fit(x, y, fit=plotflag, ax=ax1)
    mu = 10**p[0]
    siglog = p[1]
    pl, perr = gauss_fit(x, yl, fit=False)
    mu_lo = 10**pl[0]
    siglog_lo = pl[1]
    pu, perr = gauss_fit(x, yu, fit=False)
    mu_hi = 10**pu[0]
    siglog_hi = pu[1]
    ymin, ymax = get_range_pdf(lx_pdf, p, pl, pu)
    ax2.plot(x_pdf, gauss_pdf(lx_pdf, p), color=c, lw=3)
    #ax2.fill_between(x_pdf, ymin, ymax, facecolor=c, alpha=0.5, lw=0)
    logmu_ul[ir] = np.log10(mu)
    logsig_ul[ir] = siglog
    print(name1)
    print('      mean   = {0:5.2f}  (+ {1:5.2f} - {2:5.2f})'.format(mu, mu_hi-mu, mu-mu_lo))
    print('sigma_logtnu = {0:5.2f}  (+ {1:5.2f} - {2:5.2f})'.format(siglog, siglog_hi-siglog, siglog-siglog_lo))

print('-'*40)

l = ax1.legend(borderaxespad=1.2, fancybox=True, shadow=True, prop={'size': 12})
l.set_title('Class II',prop={'size':14})

plt.tight_layout()
plt.savefig(direc+'/PP7_tdisk_UL.png', dpisize=600)
plt.close()
# plt.show()


#Debris discs
#Load data with system names, masses, and ages
dd_mass = ascii.read(direc+'../Disc_Ages_and_Masses.tbl',format='ecsv',delimiter=',')
dd_name = dd_mass['System'].data
dd_age = dd_mass['Age'].data
dd_md = dd_mass['Mdust'].data
dd_ep = dd_mass['epMdust'].data
dd_em = dd_mass['emMdust'].data

agebin1 = np.where((dd_age > 10.)&(dd_age <=100.))
agebin2 = np.where((dd_age > 100.)&(dd_age <=1000.))
agebin3 = np.where((dd_age > 1000.))

tzero = 5.0 #Myr
dd_agebins = np.asarray([np.median(dd_age[agebin1]),np.median(dd_age[agebin2]),np.median(dd_age[agebin3]),tzero])
fdd = 0.25*dd_agebins

#Load radii, match radii to masses and ages
dd_raddata = ascii.read(direc+'../../emcee_resolved_disc_tablulated_values_for_modelling.csv') 
names = dd_raddata['Name'].data
radii = dd_raddata['Radius'].data

dd_rd = []

for i in range(len(dd_name)):
    argval = np.where(names == dd_name[i])
    
    dd_rd.append(radii[argval][0])

dd_rd = np.asarray(dd_rd)

#Mass extrapolation - from Matra+2025 using Wyatt+2007a
t0 = 10. #Myr
d_big = 2e-8 #Myr M_E au^-13/3
epsilon = 1.0
delta = 13./3.

tc = 100.#d_big / (dd_md * dd_rd**delta)

dd_m0 = dd_md * (1. + ((dd_age  - tzero)**epsilon)/tc)

dd_m0 = dd_m0[np.isfinite(dd_m0)]

#Do plot
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)

#x = np.linspace(0.001,50.,num=5000)

### PLOT td=age
#plt.plot(x,x,'-',color='gray',lw=1.5,zorder=15)


lx_pdf, f = np.linspace(-8,4,60), 0.3*np.array(Age)
for i, val in enumerate(Region):
    if val == 'ChaII':
        continue
    if val == 'ChaI':
        val = 'Chameleon'
    if val == 'UpperSco':
        val = 'Upper Sco'
    
    y = gauss_pdf(lx_pdf, (logmu_ul[i], logsig_ul[i]))
    ax.fill_betweenx(10**lx_pdf, Age[i]-f[i]*y, Age[i]+f[i]*y, facecolor=color[i], lw=0, zorder=20, alpha=0.3,label=val)
    

ax.errorbar(dd_age,dd_md,xerr=None,yerr=[dd_em,dd_ep],linestyle='',marker='o',mec='white',mfc='black',color='black',ms=4,alpha=0.7,zorder=21)

logmu  = np.ones(len(dd_agebins))
logsig = np.ones(len(dd_agebins))
lx_pdf = np.linspace(-6,0,100)
x_pdf = 10**lx_pdf

for ia in range(0,len(dd_agebins)):
    name1 = dd_agebins[ia]
    if ia  == 0:
        obs = dd_md[agebin1]
    elif ia == 1:
        obs = dd_md[agebin2]
    elif ia == 2:
        obs = dd_md[agebin3]
    elif ia == 3:
        obs = dd_m0
    
    det = np.ones(len(obs))
    
    x, y, yl, yu = KM_C_UL(obs,det)
    good = (yl < yu) | (yl == 0)
    x = x[good]
    y = y[good]
    yl = yl[good]
    yu = yu[good]
    
    print('-'*40)
    p, perr = gauss_fit(x, y, fit=plotflag, ax=ax)
    mu = 10**p[0]
    siglog = p[1]
    pl, perr = gauss_fit(x, yl, fit=False)
    mu_lo = 10**pl[0]
    siglog_lo = pl[1]
    pu, perr = gauss_fit(x, yu, fit=False)
    mu_hi = 10**pu[0]
    siglog_hi = pu[1]
    ymin, ymax = get_range_pdf(lx_pdf, p, pl, pu)
    #ax.plot(x_pdf, gauss_pdf(lx_pdf, p), color=c, lw=3)
    #ax.fill_between(x_pdf, ymin, ymax, facecolor=c, alpha=0.5, lw=0)
    logmu_ul[ia] = np.log10(mu)
    logsig_ul[ia] = siglog
    print('Debris discs : '+str(name1))
    print('      mean   = {0:5.4f}  (+ {1:5.4f} - {2:5.4f})'.format(mu, mu_hi-mu, mu-mu_lo))
    print('sigma_logtnu = {0:5.4f}  (+ {1:5.4f} - {2:5.4f})'.format(siglog, siglog_hi-siglog, siglog-siglog_lo))
    
    y = gauss_pdf(lx_pdf, (logmu_ul[ia], logsig_ul[ia]))
    if ia == 0:
        ax.fill_betweenx(10**lx_pdf, dd_agebins[ia]-fdd[ia]*y, dd_agebins[ia]+fdd[ia]*y, facecolor='grey', lw=0, zorder=20,label='Debris discs',alpha=0.3)
    elif ia > 0 and ia < 3:
        ax.fill_betweenx(10**lx_pdf, dd_agebins[ia]-fdd[ia]*y, dd_agebins[ia]+fdd[ia]*y, facecolor='grey', lw=0, zorder=20,alpha=0.3)
    elif ia == 3:
        ax.fill_betweenx(10**lx_pdf, dd_agebins[ia]-fdd[ia]*y, dd_agebins[ia]+fdd[ia]*y, facecolor='grey', lw=0, zorder=20,alpha=1.0)

# ax.text(0.6, -3.9, r'Oph')
# ax.text(0.85, -3.9, r'Taurus')
# ax.text(1.3, -3.9, r'Lupus')
# ax.text(1.85, -3.9, r'Cham I')
# ax.text(6.2, -3.9, r'USco')

l = ax.legend(borderaxespad=0.4, fancybox=True, shadow=False, prop={'size': 12},loc='upper right')
# l.set_title('Class II',prop={'size':14})
    

### PLOT A VISCOUS MODEL EXPECTATION
# tnu = 1. # Myr
# gamma = 1.7#9
# plt.plot(x,2*(2-gamma)*(x+tnu),'k--',lw=1.5,label=r'$t_{\nu}$ = %.1f, $\gamma$ = %.1f' % (tnu,gamma),zorder=10)

#tnu = 1. # Myr
#gamma = 1.5#9
#plt.plot(x,2*(2-gamma)*(x+tnu),'k--',lw=1.5)#,label=r'$t_{\nu}$ = %.1f, $\gamma$ = %.1f' % (tnu,gamma),zorder=10)
# plt.loglog(x,2*(2-gamma)*(x+tnu),'k-.',lw=1.5,label=r'$t_{\nu}$ = %.1f, $\gamma$ = %.1f' % (tnu,gamma),zorder=10)

# ### PLOT A MHD WIND MODEL EXPECTATION

#fm = 1

#tacc = 0.1 # Myr
#plt.plot(x,np.repeat(2*tacc*(1+fm) , len(x)),'k-.',lw=2)#,label=r'$t_{\rm acc}$ = %.1f, $f_M$ = %.1f' % (tacc,fm),zorder=10)

# tacc = 0.5 # Myr
# plt.plot(x,np.repeat(2*tacc*(1+fm) , len(x)),'g--',lw=2)#,label=r'$t_{\rm acc}$ = %.1f, $f_M$ = %.1f' % (tacc,fm),zorder=10)

#tacc = 1 # Myr
#plt.plot(x,np.repeat(2*tacc*(1+fm) , len(x)),'k-.',lw=2)#,label=r'$t_{\rm acc}$ = %.1f, $f_M$ = %.1f' % (tacc,fm),zorder=10)


# add the expectation from dust evolution
# plt.plot(np.linspace(1e5,3.e6,len(data_mdust))/1e6,100*data_mdust/earth_to_sun/data_dust_macc[:,1]/1e6, 'k--',lw=3,label='DUST')
#plt.plot(data_dust_macc[:,0],data_mdust, '--',color='orange',lw=3)#,label='DUST')


ax.set_xlabel('Region Age [Myr]', fontsize=18, labelpad=8)
ax.set_ylabel(r'$M_{\rm dust}$ [$M_{\oplus}$]', fontsize=18, labelpad=8)
# ax.set_ylim([-3.8, 3])
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_yticks([1e-5,1e-3,1e-1,1e1,1e3])
ax.set_yticklabels([r'10$^{-5}$',r'10$^{-3}$',r'10$^{-1}$',r'10$^{1}$',r'10$^{3}$'])#, fontsize=14)

ax.set_xlim([0.5 , 10000])
ax.set_xticks([1,10,100,1000,10000])
ax.set_xticklabels(['1','10', r'10$^{2}$', r'10$^{3}$',r'10$^{4}$'])#, fontsize=14)


#tw = ax.text(10,1,'MHD winds')
#tage = ax.text(10,18,r'$t_{\rm disk}\sim$age',rotation=-15,horizontalalignment='left',color='gray')
#tvi = ax.text(3,14,'Viscous',rotation=-10)
#tdu = ax.text(3,1,'Dust',color='orange')

ax.set_ylim([1e-6 , 1e3])
l = ax.legend(borderaxespad=0.8, fancybox=False, shadow=False, prop={'size': 12},loc='upper right')

# tage.set_rotation(13)
# tage.set_position([10,5])
# tvi.set_rotation(13)
# tvi.set_position([3,6])
# tdu.set_position([3,0.5])

fig.savefig(direc+'PP7_mdust_UL_violin_w_debris.pdf', bbox_inches='tight', dpi=200)


plt.show()

plt.close('all')

#References for SFR populations
for i in range(0,len(regnames)):
    print('-'*40)
    print(regnames[i])
    print(surveys[np.where(regions == regnames[i])])
    print('-'*40)
