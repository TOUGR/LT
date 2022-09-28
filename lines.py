#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:16:38 2017
@author: Alfonso Salinas
Rev. 2022/09/25
"""
import numpy as np
import matplotlib.pyplot as plt

# Constantes
pi=np.pi # Numero pi
D2r=pi/180. # Factor grados a radianes
R2d=180./pi # Factor radianes a grados

# .__doc__ para informacion

# Funciones con complejos
def C2p(x):   
    """
    Complejo a tupla de polares.
    """
    return (np.abs(x),np.angle(x))

def C2pD(x):
    """
    Complejo a tupla de polares con fase en grados.
    """
    return (np.abs(x),np.angle(x)*R2d)

def P2c(r,th):
    """
    Modulo y fase a tupla de cartesianas.
    """
    return (r*np.cos(th),r*np.sin(th))

def P2cD(r,th): # Modulo y fase en grados a tupla de cartesianas
    """
    Modulo y fase en grados a tupla de cartesianas.
    """
    thr=th*D2r
    return P2c(r,thr)


# Funciones de las lineas
def Gr(zln):
    """
    Coeficiente de reflexion en la carga.
    Impedancia de carga normalizada. 
    """
    return (zln-1)/(zln+1)

def Zwn(zpn,zln):
    """
    Impedancia de onda. Caso sin perdidas.
    (zpn,zln).
    """
    zwn=(zln+1.j*np.tan(2*pi*zpn))/(1+zln*1.j*np.tan(2*pi*zpn))
    return zwn

def distaimpe(zln):
    """
    Distancia a la carga para zw=1+jx
    (zln)
    """
    rzl=zln.real
    xzl=zln.imag
    ytan1=(xzl+np.sqrt(rzl**3-2.*rzl**2+rzl+rzl*xzl**2))/(rzl**2-rzl+xzl**2)
    ytan2=(xzl-np.sqrt(rzl**3-2.*rzl**2+rzl+rzl*xzl**2))/(rzl**2-rzl+xzl**2)
    znteo1=np.arctan(ytan1)/(2*pi)
    znteo2=np.arctan(ytan2)/(2*pi)
    if znteo1<0:
        znteo1 += 0.5
    if znteo2<0:
        znteo2 += 0.5
    return(znteo1,znteo2)

def distaadmi(zln):
    """
    Distancia a la carga para yw=1+jb
    (zln)
    """
    rzl=zln.real
    xzl=zln.imag
    ytan1=(-xzl+np.sqrt(rzl**3-2.*rzl**2+rzl+rzl*xzl**2))/(1.-rzl)
    ytan2=(-xzl-np.sqrt(rzl**3-2.*rzl**2+rzl+rzl*xzl**2))/(1.-rzl)
    znteo1=np.arctan(ytan1)/(2*pi)
    znteo2=np.arctan(ytan2)/(2*pi)
    if znteo1<0:
        znteo1 += 0.5
    if znteo2<0:
        znteo2 += 0.5
    return(znteo1,znteo2)

def lstub(z,spia):
    """
    Longitud de un stub para tener un valor de impedancia/admitancia
    normalizada z
    (z,spia)
    spia: CS AP CP AS
    Corto-Serie, Abierto-Paralelo, Corto-Paralelo, Abierto-Serie
    """
    if spia == 'CS' or spia == 'AP':
        ls=np.arctan(z)/(2.*pi)
    elif spia == 'CP' or spia == 'AS':
        ls=np.arctan(-1./z)/(2.*pi)
    else:
        print('Error en la definicion del stub')
    if ls<0:
        ls += 0.5
    return(ls)


## Funciones generales de dibujo
def Point(x,y,c='k',marker='o',s=1): # Dibuja un punto en las coordenadas cartesianas x, y
    """
    Dibuja un punto con coordenadas x,y.
    (x,y,c='k',marker='o',s=1)
    """
    plt.scatter(x,y,c=c,marker=marker,s=20*2**s)

def Arc(r,thi,thf,c='k',ls='-',lw=1): #Dibuja un arco de radio r desde thi rad hasta thf rad
    """
    Dibuja un archo de radio r desde thi rad hasta thf rad
    (r,thi,thf,c='k',ls='-',lw=1)
    """
    thes=np.linspace(thi,thf,100) # Lista de angulos 
    xsar=r*np.cos(thes)  # Proyeccion al eje x
    ysar=r*np.sin(thes)  # Proyeccion al eje y
    # Punto inicial de la flecha: desde el valor -3, dx, dy
    flecha=(xsar[-3],ysar[-3],xsar[-1]-xsar[-3],ysar[-1]-ysar[-3])
    plt.plot(xsar,ysar,c=c,ls=ls,lw=lw)
    # Se dibuja la flecha con los valores normales
    plt.arrow(*flecha,head_width=0.05, head_length=0.1,fc=c,ec=c,\
            length_includes_head=True)    

def Circle(r,c='k',ls='-',lw=1):
    """
    Dibuja un circulo de radio r
    (r,c='k',ls='-',lw=1)
    """
    thes=np.linspace(0,2.*np.pi,300)
    xsar=r*np.cos(thes)
    ysar=r*np.sin(thes)
    plt.plot(xsar,ysar,c=c,ls=ls,lw=lw)

def Line(px1,py1,px2,py2,c='k',ls='-',lw=1):
    """
    Dibuja una linea desde (px1,py1) hasta (px2,py2)
    (px1,py1,px2,py2,c='k',ls='-',lw=1)
    """
    plt.plot([px1,px2],[py1,py2],c=c,ls=ls,lw=lw)

# Funciones para la carta de Smith
def Line_r(r):
    """
    Funcion axiliar de Cir_r: Circulo de resistencia normalizada r
    (r)
    """
    rs=np.ones(3000,dtype=float)*r  # Valores de r
    xs=np.linspace(-100,100,3000,dtype=float)  # Valores de x
    zlns=rs+1.j*xs  # Impedancias normalizadas
    gas=Gr(zlns) #(impes-1)/(impes+1)   Coeficientes de reflexion
    return (gas.real,gas.imag)

def Line_x(x):
    """
    Funcion axuliar de Cir_x: Circulo de reactancia normalizada x
    (x)
    """
    xs=np.ones(3000,dtype=float)*x
    rs=np.linspace(0,100,3000,dtype=float)
    zlns=rs+1.j*xs
    gas=Gr(zlns) #(impes-1)/(impes+1)   Coeficientes de reflexion
    return (gas.real,gas.imag)

def Cir_x(x,c='k',ls='-',lw=1):
    """
    Carta de Smith: Circulo de reactancia normalizada x
    (x)
    """
    plt.plot(*Line_x(x),c=c,ls=ls,lw=lw)

def Cir_r(r,c='k',ls='-',lw=1):
    """
    Carta de Smith: Circulo de resistencia normalizada r
    (r)
    """
    plt.plot(*Line_r(r),c=c,ls=ls,lw=lw)

def Skele():
    """
    Esquema de carta de Smith
    """
    plt.axis('equal')
    plt.axis('off')
    Circle(1,c='k')
    xlis1=np.arange(-1,1,0.2) # x [-1,-0.8,-0.6,...,0.8]
    xlis2=np.arange(-5,6,dtype=int) # x [-5,-4,...,4,5]
    rlis1=np.arange(0,1,0.2)  # r [0,0.2,...0.8]
    rlis2=np.arange(0,6,dtype=int)  # r [0,1,2,...,5]
    # Dibuja los circulos anteriores
    for i in xlis1:
        Cir_x(i,c='r',lw=0.5,ls=':')
    for i in xlis2:
        Cir_x(i,c='r',lw=0.5,ls=':')
    for i in rlis1:
        Cir_r(i,c='b',lw=0.5,ls='--')
    for i in rlis2:
        Cir_r(i,c='b',lw=0.5,ls='--')
 
