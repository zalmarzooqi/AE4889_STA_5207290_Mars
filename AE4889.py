import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
#Constants Given
au= 149597870.7 #km
t_year = 365.256*24*3600 #s
gp_mars = 0.042828e6 #km3/s2
gp_sun = 132712e6 #km3/s2
mr_se = 3.0542e-6#-
mr_sm = 3.227154996101724e-7 #-
T_sm = 8253622 #s
L_sm = 208321282 #km


eps=1e-5

#Constants Taken from Literature Whenever needed 

m_s = 1.9886e30 #kg mass of sun from Wakker app B


m_mo = 7.348e22 #kg mass of moon from wakker B for verification purposes

P0 = 1367 #W/m^2 solar flux at 1 AU

PM = P0*(149597870.7/208321282)**2


c = 299792458 #m/s speed of light

print(P0/c)

#Calculated Values for use
m_m = m_s * mr_sm #kg calculated from mass ratio of mars to sun
m_e = m_s* mr_se
mu_se = m_e/(m_e+m_s) #mass parameter of sun earth

mu_sm = m_m/(m_m+m_s) #mass parameter of sun mars

mu_emo = m_mo/(m_mo+m_e) #mass parameter of earth moon

L_emo = 384401e3

r_orbit = 151111  # km
r_nd = r_orbit / L_sm

r_end= au/L_sm
print(r_nd)


#Task 1.1a

def L1Solver(mu):
    x0 = 1 - mu - (mu/3)  # good initial guess
    x1 = x0
    
    def f(x):
        r1 = x + mu
        r2 = x - (1 - mu)
        return -x + (1-mu)*r1/abs(r1)**3 + mu*r2/abs(r2)**3
    

    def df(x):
        r1 = x + mu
        r2 = x - (1 - mu)
        
        term1 = (1-mu)*(1/abs(r1)**3 - 3*r1**2/abs(r1)**5)
        term2 = mu*(1/abs(r2)**3 - 3*r2**2/abs(r2)**5)
        
        return -1 + term1 + term2

    x2 = x1 - f(x1)/df(x1)
    i = 0
    
    while abs(x2 - x1) > 1e-16:
        i += 1
        x1 = x2
        x2 = x1 - f(x1)/df(x1)

    print(-mu < x2 < 1 - mu)  # should be True for L1
    print(i)
    
    return np.array([x2, 0 ,0])

L1se = L1Solver(mu_se)
L1sm = L1Solver(mu_sm)
L1emo = L1Solver(mu_emo)
print(mu_se)
print(L1se[0])

#Task 1.1b


L1earth = (1-mu_se-L1se[0])*au
L1mars = (1-mu_sm-L1sm[0])*L_sm
L1moon = (1-mu_emo-L1emo[0])*L_emo


gammae = (1-mu_se-L1se[0])
gammam = (1-mu_sm-L1sm[0])
gammamo = (1-mu_emo-L1emo[0])
earthverf = 1.001e-2*au
marsverf= 4.748e-3*L_sm
moonverf= 1.509e-1*L_emo

perdifearth = 100*(abs(L1earth-earthverf)/earthverf)
perdifmars =100*( abs(L1mars-marsverf)/marsverf)
perdifmoon =100*( abs(L1moon-moonverf)/moonverf)

#task 1.2
def acc(mu,r):
    r1 = r+np.array([mu, 0,0])
    r2 = r+np.array([-(1-mu), 0, 0])
    w=np.array([0, 0, 1])
    a_s= -((1-mu)/(np.linalg.norm(r1))**3)*r1
    a_b = (-mu/(np.linalg.norm(r2))**3)*r2
    a_c =- np.cross(w,np.cross(w,r))
    a= a_s+a_b+a_c
    return np.vstack((a_s, a_b, a_c, a))

accL1E = acc(mu_se,L1se)

accL1M = acc(mu_sm,L1sm)


accdiffsun = 100*(abs(accL1E[0,0]-accL1M[0,0])/accL1M[0,0])
accdiffbody =100*(abs( accL1E[1,0]-accL1M[1,0])/accL1M[1,0])
accdiffcentrip= 100*(abs( accL1E[2,0]-accL1M[2,0])/accL1M[2,0])
    
#artifical Langrage Points


dx= -0.00050

dvec = np.array([dx,0,0])

AL1se = L1se+ dvec
AL1sm = L1sm+dvec

#task 2 Artifical Langrange points

#solar sail orientation

def nhat(mu,r):
    
    r1 = r + np.array([mu, 0, 0])
    r2 = r + np.array([-(1-mu), 0, 0])

    Ux = -r[0] +((1-mu)*r1[0]/np.linalg.norm(r1)**3) + (mu*r2[0]/np.linalg.norm(r2)**3)
    Uy =- r[1] + ((1-mu)*r1[1]/np.linalg.norm(r1)**3) + (mu*r2[1]/np.linalg.norm(r2)**3)
    Uz =  r[2]*(((1-mu)/np.linalg.norm(r1)**3) +( mu/np.linalg.norm(r2)**3))
   
    gradU = np.array([Ux, Uy, Uz])
    return gradU/np.linalg.norm(gradU)


nse= nhat(mu_se,AL1se)
nsm= nhat(mu_sm,AL1sm)
 # lightness number
def beta(mu,r,n):
    r1 = r + np.array([mu, 0, 0])
    r2 = r + np.array([-(1-mu), 0, 0])
    r1hat = r1/np.linalg.norm(r1)

    Ux = -r[0] +((1-mu)*r1[0]/np.linalg.norm(r1)**3) + (mu*r2[0]/np.linalg.norm(r2)**3)
    Uy =- r[1] + ((1-mu)*r1[1]/np.linalg.norm(r1)**3) + (mu*r2[1]/np.linalg.norm(r2)**3)
    Uz =  r[2]*(((1-mu)/np.linalg.norm(r1)**3) +( mu/np.linalg.norm(r2)**3))
   
    gradU = np.array([Ux, Uy, Uz])
    dot_gradU_n = np.dot(gradU, n)
    dot_r1hat_n = np.dot(r1hat, n)

    beta = (np.linalg.norm(r1)**2 * dot_gradU_n) / ((1 - mu) * dot_r1hat_n**2)
    return beta

betase= beta(mu_se,AL1se,nse)
betasm= beta(mu_sm,AL1sm,nsm)

#side length
m_sc = 100
lse =np.sqrt((betase*c*gp_sun*((10**3)**3)*m_sc)/(P0*(au*10**(3))**2))

lsm =np.sqrt((betasm*c*gp_sun*((10**3)**3)*m_sc)/(PM*(au*10**(3))**2))


#task 2.4

ALVAR= np.arange(dx, 2*dx+(dx/100),(dx/100))
betasEARRAY = []
betasMARRAY = []
for dxi in ALVAR:
    dvector = np.array([dxi,0,0])

    AL1seVAR = L1se+ dvector
    AL1smVAR = L1sm+ dvector
    nse= nhat(mu_se,AL1seVAR)
    nsm= nhat(mu_sm,AL1smVAR)
    betasE= beta(mu_se,AL1seVAR,nse)
    betasM = beta(mu_sm,AL1smVAR,nsm)
    betasEARRAY.append(betasE)
    betasMARRAY.append(betasM)
    
betasEARRAY = np.array(betasEARRAY)
betasMARRAY = np.array(betasMARRAY)

plt.figure(figsize=(7, 5))
plt.plot(ALVAR, betasEARRAY)
plt.xlabel("Shift from L1, Δx [-]")
plt.ylabel("Lightness number β [-]")
plt.title("Spacecraft lightness number with respect to Sun–Earth Artificial L1 point")
plt.grid(True)
plt.show()


plt.figure(figsize=(7, 5))
plt.plot(ALVAR, betasMARRAY)
plt.xlabel("Shift from L1, Δx [-]")
plt.ylabel("Lightness number β [-]")
plt.title("Spacecraft lightness number with respect to Sun–Mars Artificial L1 point")
plt.grid(True)
plt.show()

#task 3.1a








def EigVal(mu,r,al=False,beta=0):   
    r1 = r+np.array([mu, 0,0])
    r2 = r+np.array([-(1-mu), 0, 0])
        
    def Uxx(r1,r2):
        
        
        term1 = (1-mu)*(1/np.linalg.norm(r1)**3 - 3*r1[0]**2/np.linalg.norm(r1)**5)
        term2 = mu*(1/np.linalg.norm(r2)**3 - 3*r2[0]**2/np.linalg.norm(r2)**5)
        
        return -1 + term1 + term2
    #print('Uxx' , Uxx(r))
    def Uxy(r1,r2):
     
       term1 = (1-mu)*(-3*r1[0]*r1[1]/np.linalg.norm(r1)**5)
       term2 = mu*(- 3*r2[0]*r2[1]/np.linalg.norm(r2)**5)
       return term1 + term2
    #print('Uxy' , Uxy(r))
    def Uyy(r1,r2):
      
      term1 = (1-mu)*(1/np.linalg.norm(r1)**3 - 3*r1[1]**2/np.linalg.norm(r1)**5)
      term2 = mu*(1/np.linalg.norm(r2)**3 - 3*r2[1]**2/np.linalg.norm(r2)**5)
      
      return -1 + term1 + term2
    print('Uxx' , Uxx(r1,r2))
    print('Uyx' , Uxy(r1,r2))
    print('Uyy' , Uyy(r1,r2))
    if al ==True:
        beta =beta
    else:
        beta =0

    asx= -2*beta*(1-mu)*(r1[0])/np.linalg.norm(r1)**2    
    print(asx)
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-Uxx(r1,r2)+asx,-Uxy(r1,r2), 0, 2],
        [-Uxy(r1,r2),- Uyy(r1,r2), -2, 0]
        ])
    print(A)
    a=1 
    b=4+Uxx(r1,r2)+Uyy(r1,r2)
    c= Uxx(r1,r2)*Uyy(r1,r2)
    disc = np.sqrt(b**2 + 4*a*c + 0j)
    x1 = (-b + disc) / (2*a)
    x2 = (-b - disc) / (2*a)

    eigvals = np.array([
    np.sqrt(x1),
    -np.sqrt(x1),
    np.sqrt(x2),
    -np.sqrt(x2)
    ])
    return np.linalg.eig(A)

L1seval,L1sevec = EigVal(mu_se,L1se)

AL1seval ,AL1sevec =EigVal(mu_se,AL1se,True,betase)

L1smval, L1smvec =EigVal(mu_sm,L1sm)


AL1smval, AL1smvec=EigVal(mu_sm,AL1sm,True,betasm)

L1emoval, L1emovec =EigVal(mu_emo,L1emo)

print('Eigenvales of Sun Earth Lagrange Point' , L1seval)
#print('Eigenvalues of Artificial Sun Earth Lagrange Point' , AL1seval)
#print('Eigenvalues of Sun Mars Lagrange Point' , L1smval)
#print('Eigenvalues of Artifical Sun Mars Lagrange Point' , AL1smval)


print('Eigenvectors of Sun Mars Lagrange Point' , L1smvec)

print('Eigenvectors of Earth Moon Lagrange Point' , L1emovec)



#Task 4.1



XL1sm = np.hstack([L1sm[:2], [0, 0]])
XL1emo = np.hstack([L1emo[:2], [0, 0]])



#stable manifold start point


X0ps=XL1sm +eps*(L1smvec[:,0]/np.linalg.norm(L1smvec[:,0]))
X0ms=XL1sm -eps*(L1smvec[:,0]/np.linalg.norm(L1smvec[:,0]))
#unstable manifold start point

X0pu=XL1sm +eps*(L1smvec[:,1]/np.linalg.norm(L1smvec[:,1]))
X0mu=XL1sm -eps*(L1smvec[:,1]/np.linalg.norm(L1smvec[:,1]))




t_years=5

t_nd = t_years * (2*np.pi / 1.88085)


def cr3bp(t, X,mu):
    x= X[0]
    y= X[1]
    vx= X[2]
    vy= X[3]
    
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2)
    
    ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-(1-mu))/r2**3
    ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
    
    return [vx, vy, ax, ay]


def cr3bp_sail(t, X, mu, beta, alpha_deg):
    x, y, vx, vy = X

    r1_vec = np.array([x + mu, y])
    r2_vec = np.array([x - (1 - mu), y])

    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    ax_cr3bp = 2*vy + x - (1-mu)*r1_vec[0]/r1**3 - mu*r2_vec[0]/r2**3
    ay_cr3bp = -2*vx + y - (1-mu)*r1_vec[1]/r1**3 - mu*r2_vec[1]/r2**3

    r1_hat = r1_vec / r1

    alpha = np.deg2rad(alpha_deg)

    n_hat = np.array([
        np.cos(alpha)*r1_hat[0] - np.sin(alpha)*r1_hat[1],
        np.sin(alpha)*r1_hat[0] + np.cos(alpha)*r1_hat[1]
    ])

    a_sail = beta * (1 - mu) / r1**2 * (np.dot(r1_hat, n_hat))**2 * n_hat

    ax = ax_cr3bp + a_sail[0]
    ay = ay_cr3bp + a_sail[1]

    return [vx, vy, ax, ay]



# Time spans
t_forward = (0, t_nd)
t_backward = (0, -t_nd)

# Propagate all 4 branches
sol_sp = solve_ivp(cr3bp, t_backward, X0ps, args=(mu_sm,), rtol=1e-12, atol=1e-12)
sol_sm = solve_ivp(cr3bp, t_backward, X0ms, args=(mu_sm,), rtol=1e-12, atol=1e-12)

sol_up = solve_ivp(cr3bp, t_forward, X0pu, args=(mu_sm,), rtol=1e-12, atol=1e-12)
sol_um = solve_ivp(cr3bp, t_forward, X0mu, args=(mu_sm,), rtol=1e-12, atol=1e-12)


# Mars position in rotating frame
mars = np.array([1 - mu_sm, 0.0])
sun = np.array([-mu_sm, 0.0])

theta = np.linspace(0, 2*np.pi, 400)

orbit_x = mars[0] + r_nd * np.cos(theta)
orbit_y = mars[1] + r_nd * np.sin(theta)

theta2 = np.linspace(0, 2*np.pi, 400)

orbit_x2 = sun[0] + r_end * np.cos(theta)
orbit_y2 = sun[1] + r_end * np.sin(theta)


def first_intersection_state(sol, planet, r_nd):
    X = sol.y.T          
    t = sol.t

    rho = np.sqrt((X[:, 0] - planet[0])**2 + (X[:, 1] - planet[1])**2)
    g = rho - r_nd

    crossings = np.where(g[:-1] * g[1:] < 0)[0]

    if len(crossings) == 0:
        return None

    i = crossings[0]


    alpha = -g[i] / (g[i+1] - g[i])

    X_int = X[i] + alpha * (X[i+1] - X[i])
    t_int = t[i] + alpha * (t[i+1] - t[i])

    return X_int, t_int

X_cap_um, t_cap_um = first_intersection_state(sol_um, mars, r_nd)


x_cap  = X_cap_um[0]
y_cap  = X_cap_um[1]
vx_cap = X_cap_um[2]
vy_cap = X_cap_um[3]

print("Capture state:", X_cap_um)
print("Capture time:", t_cap_um* ( 1.88085/(2*np.pi)))


def capture_delta_v(X_cap, mars, mu, L_dim, T_dim):
    x, y, vx, vy = X_cap


    r_rel = np.array([x - mars[0], y - mars[1]])
    r = np.linalg.norm(r_rel)

 
    v_man_rot = np.array([vx, vy])



    omega_cross_r = np.array([-r_rel[1], r_rel[0]])
    v_man_rel = v_man_rot + omega_cross_r

 
    v_circ_mag = np.sqrt(mu / r)


    t_hat_pro = np.array([-r_rel[1], r_rel[0]]) / r
    t_hat_ret = -t_hat_pro

    v_circ_pro = v_circ_mag * t_hat_pro
    v_circ_ret = v_circ_mag * t_hat_ret

    dv_pro_nd = np.linalg.norm(v_circ_pro - v_man_rel)
    dv_ret_nd = np.linalg.norm(v_circ_ret - v_man_rel)


    V_dim = L_dim / T_dim

    dv_pro_kms = dv_pro_nd * V_dim
    dv_ret_kms = dv_ret_nd * V_dim

    return dv_pro_kms, dv_ret_kms

def capture_delta_v_sun(X_cap, sun, L_dim, T_dim, gp_sun):
    x, y, vx, vy = X_cap


    r_rel_nd = np.array([x - sun[0], y - sun[1]])
    r_nd = np.linalg.norm(r_rel_nd)

    v_rot_nd = np.array([vx, vy])


    omega_cross_r_nd = np.array([-r_rel_nd[1], r_rel_nd[0]])
    v_rel_nd = v_rot_nd + omega_cross_r_nd


    V_dim = L_dim / T_dim
    v_man_kms = v_rel_nd * V_dim


    r_km = r_nd * L_dim


    v_circ_mag = np.sqrt(gp_sun / r_km)


    t_hat_pro = np.array([-r_rel_nd[1], r_rel_nd[0]]) / r_nd

    v_circ_pro = v_circ_mag * t_hat_pro

    dv = np.linalg.norm(v_circ_pro - v_man_kms)
    
    return dv


dv_pro, dv_ret = capture_delta_v(
    X_cap_um,
    mars,
    mu_sm,
    L_sm,
    T_sm
)

print("Delta-V prograde  [m/s]:", dv_pro*1000)
print("Delta-V retrograde [m/s]:", dv_ret*1000)
print("Minimum Delta-V   [m/s]:", min(dv_pro, dv_ret)*1000)


plt.figure(figsize=(8, 6))

# Manifolds
plt.plot(sol_sp.y[0], sol_sp.y[1], 'b', linewidth=0.8,label ='stable right')
plt.plot(sol_sm.y[0], sol_sm.y[1], 'b--', linewidth=0.8,label ='stable left')
plt.plot(sol_up.y[0], sol_up.y[1], 'r', linewidth=0.8,label ='unstable left')
plt.plot(sol_um.y[0], sol_um.y[1], 'r--', linewidth=0.8,label ='unstable right')

plt.plot(orbit_x2, orbit_y2, 'g', linewidth=2, label='Earth Orbit')
# Mars-centered orbit
plt.plot(orbit_x, orbit_y, 'g', linewidth=2, label='Target orbit')
# Bodies
plt.scatter(sun[0], sun[1], color='orange', s=100, label='Sun')
plt.scatter(mars[0], mars[1], color='red', s=100, label='Mars')
plt.scatter(XL1sm[0], XL1sm[1], color='k', s=60, label='L1')

plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel("x (CR3BP)")
plt.ylabel("y (CR3BP)")
plt.title("System-wide View: CR3BP Manifolds (Sun–Mars)")

plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(7, 7))

# Manifolds (same data, but visually important near Mars)
plt.plot(sol_sp.y[0], sol_sp.y[1], 'b', alpha=0.6,label ='stable right')
plt.plot(sol_sm.y[0], sol_sm.y[1], 'b--', alpha=0.6,label ='stable left')
plt.plot(sol_up.y[0], sol_up.y[1], 'r', alpha=0.6,label ='unstable left')
plt.plot(sol_um.y[0], sol_um.y[1], 'r--', alpha=0.6,label ='unstable right')
# Mars-centered orbit
plt.plot(orbit_x, orbit_y, 'g', linewidth=2, label='Target orbit')

# Focus point: Mars
plt.scatter(mars[0], mars[1], color='red', s=100, label='Mars')

# L1 (still visible nearby structure)
plt.scatter(XL1sm[0], XL1sm[1], color='k', s=50, label='L1')

# ---- zoom limits ----
x_all = np.concatenate([
    sol_sp.y[0], sol_sm.y[0], sol_up.y[0], sol_um.y[0],
    orbit_x,
    [mars[0], XL1sm[0]]
])

y_all = np.concatenate([
    sol_sp.y[1], sol_sm.y[1], sol_up.y[1], sol_um.y[1],
    orbit_y,
    [mars[1], XL1sm[1]]
])

# Only keep points near Mars/L1 region to avoid zooming out too much
x_min_region = min(mars[0], XL1sm[0]) - 2*r_nd
x_max_region = max(mars[0], XL1sm[0]) + 2*r_nd
y_min_region = -2*r_nd
y_max_region =  2*r_nd

mask = (
    (x_all >= x_min_region) & (x_all <= x_max_region) &
    (y_all >= y_min_region) & (y_all <= y_max_region)
)

x_zoom = x_all[mask]
y_zoom = y_all[mask]

x_center = 0.5 * (np.max(x_zoom) + np.min(x_zoom))
y_center = 0.5 * (np.max(y_zoom) + np.min(y_zoom))

x_span = np.max(x_zoom) - np.min(x_zoom)
y_span = np.max(y_zoom) - np.min(y_zoom)

span = 1.2 * max(x_span, y_span)

plt.xlim(x_center - span/2, x_center + span/2)
plt.ylim(y_center - span/2, y_center + span/2)

plt.gca().set_aspect('equal', adjustable='box')


plt.xlabel("x (CR3BP)")
plt.ylabel("y (CR3BP)")
plt.title("Local View: Mars Orbit + Nearby Manifolds")

plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
'''
plt.scatter(
    x_cap,
    y_cap,
    color='magenta',
    s=120,
    marker='x',
    linewidths=3,
    label='Capture Point'
)
'''

# Find index in sol_um closest to the intersection time
i_cap = np.argmin(np.abs(sol_um.t - t_cap_um))

# Manifold segment from L1 to intersection
x_seg = sol_um.y[0, :i_cap+1]
y_seg = sol_um.y[1, :i_cap+1]

plt.figure(figsize=(7, 7))

# Intersecting manifold only
plt.plot(
    x_seg, y_seg,
    'r--',
    linewidth=2,
    label='Intersecting unstable manifold'
)

# Mars-centered target orbit
plt.plot(
    orbit_x, orbit_y,
    'g',
    linewidth=2,
    label='Target Mars orbit'
)

# Mars
plt.scatter(
    mars[0], mars[1],
    color='red',
    s=100,
    label='Mars',
    zorder=5
)

# L1 point
plt.scatter(
    XL1sm[0], XL1sm[1],
    color='black',
    s=70,
    label='Sun–Mars L1',
    zorder=6
)

# Capture/intersection point
plt.scatter(
    X_cap_um[0], X_cap_um[1],
    color='magenta',
    s=120,
    marker='x',
    linewidths=3,
    label='Capture point',
    zorder=7
)

# Zoom around L1, Mars orbit, and manifold segment
x_all = np.concatenate([
    x_seg,
    orbit_x,
    [mars[0], XL1sm[0], X_cap_um[0]]
])

y_all = np.concatenate([
    y_seg,
    orbit_y,
    [mars[1], XL1sm[1], X_cap_um[1]]
])

x_center = 0.5 * (np.max(x_all) + np.min(x_all))
y_center = 0.5 * (np.max(y_all) + np.min(y_all))

x_span = np.max(x_all) - np.min(x_all)
y_span = np.max(y_all) - np.min(y_all)

span = 1.2 * max(x_span, y_span)

plt.xlim(x_center - span/2, x_center + span/2)
plt.ylim(y_center - span/2, y_center + span/2)

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("x [-]")
plt.ylabel("y [-]")
plt.title("Close View: L1 to Mars-Orbit Capture Point")
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()


alphas = np.arange(-90, 90 + 2.5, 2.5)


beta = 0.025
viable_transfers = []

for alpha_deg in alphas:

    sol = solve_ivp(
        cr3bp_sail,
        t_backward,
        X0ms,
        args=(mu_sm, beta, alpha_deg),
        rtol=1e-12,
        atol=1e-12
    )

    result = first_intersection_state(sol, sun, r_end)

    if result is None:
        continue

    X_int, t_int = result

    dv_min = capture_delta_v_sun(
        X_int,
        sun,
        L_sm,
        T_sm,
        gp_sun
    )

    

    # Positive transfer duration in Earth years
    transfer_time = -t_int * T_sm / t_year

    viable_transfers.append({
        "alpha_deg": alpha_deg,
        "time_nd": t_int,
        "delta_v_ms": dv_min*1000,
        "time_years": transfer_time,
        "state": X_int,
        "solution": sol
    })
for case in viable_transfers:
    print(
        f"alpha = {case['alpha_deg']:6.1f} deg | "
        f"time = {case['time_years']:8.3f} yr | "
        f"dV = {case['delta_v_ms']:8.3f} m/s"
    )
best_earth_case = min(
    viable_transfers,
    key=lambda x: x["delta_v_ms"]
)

print("\nOPTIMAL EARTH-L1 TRANSFER")
print("----------------")
print("Cone angle [deg]:", best_earth_case["alpha_deg"])
print("Transfer time [yr]:", best_earth_case["time_years"])
print("Minimum Delta-V [m/s]:", best_earth_case["delta_v_ms"])


viable_mars_transfers = []

for alpha_deg in alphas:
    sol = solve_ivp(
        cr3bp_sail,
        t_forward,
        X0mu,
        args=(mu_sm, beta, alpha_deg),
        rtol=1e-12,
        atol=1e-12,
    )

    result = first_intersection_state(sol, mars, r_nd)

    if result is None:
        continue

    X_int, t_int = result

    dv_pro, dv_ret = capture_delta_v(
        X_int,
        mars,
       mu_sm,
        L_sm,
        T_sm
    )

    dv_min_ms = min(dv_pro, dv_ret) * 1000
    transfer_time_years = t_int * T_sm / t_year

    viable_mars_transfers.append({
        "alpha_deg": alpha_deg,
        "time_nd": t_int,
        "time_years": transfer_time_years,
        "delta_v_ms": dv_min_ms,
        "state": X_int,
        "solution": sol,
        
    })
    
for case in viable_mars_transfers:
    print(
        f"alpha = {case['alpha_deg']:6.1f} deg | "
        f"time = {case['time_years']:8.3f} yr | "
        f"dV = {case['delta_v_ms']:8.3f} m/s"
    )

best_mars_case = min(
    viable_mars_transfers,
    key=lambda x: x["delta_v_ms"]
)

print("\nOPTIMAL L1-MARS TRANSFER")
print("---------------------")
print("Cone angle [deg]:", best_mars_case["alpha_deg"])
print("Transfer time [yr]:", best_mars_case["time_years"])
print("Minimum Delta-V [m/s]:", best_mars_case["delta_v_ms"])

# Get optimal solutions and intersection states/times
best_leg1_sol = best_earth_case["solution"]   # backward: Earth orbit -> L1
best_leg2_sol = best_mars_case["solution"]    # forward: L1 -> Mars orbit

X_earth_int = best_earth_case["state"]
X_mars_int  = best_mars_case["state"]

t_earth_int = best_earth_case["time_nd"]      # store this if not already stored
t_mars_int  = best_mars_case["time_nd"]       # store this if not already stored
# ============================================================
# Trim optimal trajectories exactly at their intersection points
# ============================================================

# Leg 1 was propagated backward, so t goes from 0 to negative
mask_leg1 = best_leg1_sol.t >= t_earth_int

x_leg1 = np.append(best_leg1_sol.y[0, mask_leg1], X_earth_int[0])
y_leg1 = np.append(best_leg1_sol.y[1, mask_leg1], X_earth_int[1])

# Leg 2 was propagated forward, so t goes from 0 to positive
mask_leg2 = best_leg2_sol.t <= t_mars_int

x_leg2 = np.append(best_leg2_sol.y[0, mask_leg2], X_mars_int[0])
y_leg2 = np.append(best_leg2_sol.y[1, mask_leg2], X_mars_int[1])


# ============================================================
# SYSTEM-WIDE VIEW
# ============================================================

plt.figure(figsize=(8,8))

# Optimal transfer legs, trimmed
plt.plot(
    x_leg1, y_leg1,
    'b',
    linewidth=1.2,
    label=f"Optimal Earth→L1, α={best_earth_case['alpha_deg']}°"
)

plt.plot(
    x_leg2, y_leg2,
    'r',
    linewidth=1.2,
    label=f"Optimal L1→Mars, α={best_mars_case['alpha_deg']}°"
)

# Earth orbit around Sun
plt.plot(
    orbit_x2, orbit_y2,
    'g',
    linewidth=2,
    label='Earth orbit'
)

# Martian parking orbit
plt.plot(
    orbit_x, orbit_y,
    'm',
    linewidth=2,
    label='Martian parking orbit'
)

# Bodies and L1
plt.scatter(
    sun[0], sun[1],
    color='orange',
    s=100,
    label='Sun'
)

plt.scatter(
    mars[0], mars[1],
    color='red',
    s=100,
    label='Mars'
)

plt.scatter(
    XL1sm[0], XL1sm[1],
    color='k',
    s=60,
    label='Sun–Mars L1'
)

# Intersection points
plt.scatter(
    X_earth_int[0], X_earth_int[1],
    color='cyan',
    s=100,
    marker='x',
    linewidths=3,
    label='Earth departure point'
)

plt.scatter(
    X_mars_int[0], X_mars_int[1],
    color='magenta',
    s=100,
    marker='x',
    linewidths=3,
    label='Mars capture point'
)

plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel("x [-]")
plt.ylabel("y [-]")
plt.title("System-wide View: Optimal Solar-Sail Earth–Mars Transfer")

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# CLOSE-UP VIEW: MARS CAPTURE INCLUDING L1
# ============================================================

plt.figure(figsize=(8, 6))

# Only optimal L1 -> Mars leg, trimmed at capture point
plt.plot(
    x_leg2, y_leg2,
    'r',
    linewidth=1.4,
    label=f"Optimal L1→Mars, α={best_mars_case['alpha_deg']}°"
)

# Martian parking orbit
plt.plot(
    orbit_x, orbit_y,
    'm',
    linewidth=2,
    label='Martian parking orbit'
)

# Mars, L1, and capture point
plt.scatter(
    mars[0], mars[1],
    color='red',
    s=100,
    label='Mars'
)

plt.scatter(
    XL1sm[0], XL1sm[1],
    color='k',
    s=60,
    label='Sun–Mars L1'
)

plt.scatter(
    X_mars_int[0], X_mars_int[1],
    color='magenta',
    s=120,
    marker='x',
    linewidths=3,
    label='Mars capture point'
)

# Automatically choose limits so L1 and Mars orbit are visible
x_all = np.concatenate([
    x_leg2,
    orbit_x,
    [mars[0], XL1sm[0], X_mars_int[0]]
])

y_all = np.concatenate([
    y_leg2,
    orbit_y,
    [mars[1], XL1sm[1], X_mars_int[1]]
])

x_center = 0.5*(np.max(x_all) + np.min(x_all))
y_center = 0.5*(np.max(y_all) + np.min(y_all))

x_span = np.max(x_all) - np.min(x_all)
y_span = np.max(y_all) - np.min(y_all)

# Use the larger span for both axes
span = 1.1 * max(x_span, y_span)

plt.xlim(
    x_center - span/2,
    x_center + span/2
)

plt.ylim(
    y_center - span/2,
    y_center + span/2
)

plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel("x [-]")
plt.ylabel("y [-]")
plt.title("Close-up View: Optimal Mars Capture")

plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()



# ============================
# Earth → L1 ΔV vs cone angle
# ============================

earth_alpha = np.array([case["alpha_deg"] for case in viable_transfers])
earth_dv = np.array([case["delta_v_ms"] for case in viable_transfers])

plt.figure(figsize=(8, 5))
plt.plot(earth_alpha, earth_dv, marker='o', linestyle='-')

plt.xlim(-90, 90)
plt.xlabel("Cone angle α [deg]")
plt.ylabel("Earth → L1 ΔV [m/s]")
plt.title("ΔV at Earth’s orbit with Respect to Cone Angle")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================
# L1 → Mars ΔV vs cone angle
# ============================

mars_alpha = np.array([case["alpha_deg"] for case in viable_mars_transfers])
mars_dv = np.array([case["delta_v_ms"] for case in viable_mars_transfers])

plt.figure(figsize=(8, 5))
plt.plot(mars_alpha, mars_dv, marker='s', linestyle='-')

plt.xlim(-90, 90)
plt.xlabel("Cone angle α [deg]")
plt.ylabel("L1 → Mars ΔV [m/s]")
plt.title("ΔV at Martian parking orbit with Respect to Cone Angle")
plt.grid(True)
plt.tight_layout()
plt.show()

earth_dv_dict = {
    case["alpha_deg"]: case["delta_v_ms"]
    for case in viable_transfers
}

mars_dv_dict = {
    case["alpha_deg"]: case["delta_v_ms"]
    for case in viable_mars_transfers
}

common_alphas = sorted(set(earth_dv_dict.keys()) & set(mars_dv_dict.keys()))

total_dv = np.array([
    earth_dv_dict[a] + mars_dv_dict[a]
    for a in common_alphas
])

plt.figure(figsize=(8, 5))
plt.plot(common_alphas, total_dv, marker='^', linestyle='-')

plt.xlim(-90, 90)
plt.xlabel("Cone angle α [deg]")
plt.ylabel("Total ΔV [m/s]")
plt.title("Total Earth → L1 → Mars ΔV with Respect to Cone Angle")
plt.grid(True)
plt.tight_layout()
plt.show()

#task 4,5



phi_0 = 122.35   # deg, Earth ahead of Mars

n_earth = 360.0              # deg/year
n_mars  = 360.0 / 1.88085    # deg/year
n_rel   = n_earth - n_mars   # deg/year


start_date = datetime(2030, 1, 1)

X_earth_int = np.real(best_earth_case["state"])

r_earth_rel_sun = np.array([
    X_earth_int[0] - sun[0],
    X_earth_int[1] - sun[1]
])

phi_req = np.degrees(np.arctan2(
    r_earth_rel_sun[1],
    r_earth_rel_sun[0]
)) % 360


delta_phi_backward = (phi_0 - phi_req) % 360

wait_years_backward = delta_phi_backward / n_rel

launch_date_before = start_date - timedelta(
    days=wait_years_backward * 365.256
)

print("Earliest launch date before 2030:", launch_date_before.date())




    