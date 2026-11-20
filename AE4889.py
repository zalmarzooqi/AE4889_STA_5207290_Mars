import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
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

L_emo = 384401  # km, consistent with the other length units

r_orbit = 368888  # km
r_nd = r_orbit / L_sm

r_end= au/L_sm
print(r_nd)


#Task 1.1a

def L2Solver(mu):
    x0 = 1 - mu + (mu/3)**(1/3)  # good initial guess
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

    for i in range(100):
        x2 = x1 - f(x1)/df(x1)
        if abs(x2-x1) <= 1e-14:
            break
        x1 = x2
    else:
        raise RuntimeError("L2 Newton solver did not converge")
    if not x2 > 1-mu:
        raise RuntimeError("Newton solver converged outside the L2 region")
    print("L2 Newton iterations:", i+1)

    return np.array([x2, 0 ,0])

L2se = L2Solver(mu_se)
L2sm = L2Solver(mu_sm)
L2emo = L2Solver(mu_emo)
print(mu_se)
print(L2se[0])

#Task 1.1b


L2earth = (L2se[0] - (1-mu_se))*au
L2mars = (L2sm[0] - (1-mu_sm))*L_sm
L2moon = (L2emo[0] - (1-mu_emo))*L_emo


gammae = L2se[0] - (1-mu_se)
gammam = L2sm[0] - (1-mu_sm)
gammamo = L2emo[0] - (1-mu_emo)
# Independent bracketed root verification; the old reference values were for L1.
def verify_l2(mu):
    def equilibrium(x):
        return (x - (1-mu)*(x+mu)/abs(x+mu)**3
                - mu*(x-(1-mu))/abs(x-(1-mu))**3)
    return brentq(equilibrium, 1-mu+1e-8, 2.0, xtol=1e-14)

earthverf = (verify_l2(mu_se) - (1-mu_se))*au
marsverf = (verify_l2(mu_sm) - (1-mu_sm))*L_sm
moonverf = (verify_l2(mu_emo) - (1-mu_emo))*L_emo
perdifearth = 100*abs(L2earth-earthverf)/earthverf
perdifmars = 100*abs(L2mars-marsverf)/marsverf
perdifmoon = 100*abs(L2moon-moonverf)/moonverf
print("L2 distance verification errors [%]:", perdifearth, perdifmars, perdifmoon)

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

accL2E = acc(mu_se,L2se)

accL2M = acc(mu_sm,L2sm)


accdiffsun = 100*(abs(accL2E[0,0]-accL2M[0,0])/abs(accL2M[0,0]))
accdiffbody =100*(abs( accL2E[1,0]-accL2M[1,0])/abs(accL2M[1,0]))
accdiffcentrip= 100*(abs( accL2E[2,0]-accL2M[2,0])/abs(accL2M[2,0]))
    
#artifical Langrage Points


dx= -0.001

dvec = np.array([dx,0,0])

AL2se = L2se + dvec
AL2sm = L2sm + dvec
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


nse= nhat(mu_se,AL2se)
nsm= nhat(mu_sm,AL2sm)
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

betase= beta(mu_se,AL2se,nse)
betasm= beta(mu_sm,AL2sm,nsm)

#side length
m_sc = 100
lse = np.sqrt(
    betase * c * gp_sun * 1e9 * m_sc
    / (2 * P0 * (au * 1e3)**2)
)

lsm = np.sqrt(
    betasm * c * gp_sun * 1e9 * m_sc
    / (2 * P0 * (au * 1e3)**2)
)


#task 2.4

ALVAR= np.arange(dx, 2*dx+(dx/100),(dx/100))
betasEARRAY = []
betasMARRAY = []
for dxi in ALVAR:
    dvector = np.array([dxi,0,0])

    AL2seVAR = L2se+ dvector
    AL2smVAR = L2sm+ dvector
    nse_var= nhat(mu_se,AL2seVAR)
    nsm_var= nhat(mu_sm,AL2smVAR)
    betasE= beta(mu_se,AL2seVAR,nse_var)
    betasM = beta(mu_sm,AL2smVAR,nsm_var)
    betasEARRAY.append(betasE)
    betasMARRAY.append(betasM)
    
betasEARRAY = np.array(betasEARRAY)
betasMARRAY = np.array(betasMARRAY)

plt.figure(figsize=(7, 5))
plt.plot(L2se[0] + ALVAR, betasEARRAY, label="Sun-Earth")
plt.legend()
plt.xlabel("Artificial L2 x-coordinate [-]")
plt.ylabel("Lightness number β [-]")
plt.title("Spacecraft lightness number with respect to Sun–Earth Artificial L2 point")
plt.grid(True)
plt.show()


plt.figure(figsize=(7, 5))
plt.plot(L2sm[0] + ALVAR, betasMARRAY, label="Sun-Mars")
plt.legend()
plt.xlabel("Artificial L2 x-coordinate [-]")
plt.ylabel("Lightness number β [-]")
plt.title("Spacecraft lightness number with respect to Sun–Mars Artificial L2 point")
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

    # Constant cone angle alpha=0: sail normal follows the Sun-spacecraft line,
    # matching cr3bp_sail. These derivatives are evaluated on the x-axis.
    sail_gradient = beta*(1-mu)/np.linalg.norm(r1)**3
    asx = -2*sail_gradient
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-Uxx(r1,r2)+asx, -Uxy(r1,r2), 0, 2],
        [-Uxy(r1,r2), -Uyy(r1,r2)+sail_gradient, -2, 0]
    ])
    return np.linalg.eig(A)

L2seval,L2sevec = EigVal(mu_se,L2se)

AL2seval ,AL2sevec =EigVal(mu_se,AL2se,True,betase)

L2smval, L2smvec =EigVal(mu_sm,L2sm)


AL2smval, AL2smvec=EigVal(mu_sm,AL2sm,True,betasm)

L2emoval, L2emovec =EigVal(mu_emo,L2emo)

print('Eigenvalues of Sun Earth Lagrange Point' , L2seval)
print('Eigenvalues of Artificial Sun Earth Lagrange Point' , AL2seval)
print('Eigenvalues of Sun Mars Lagrange Point' , L2smval)
print('Eigenvalues of Artifical Sun Mars Lagrange Point' , AL2smval)


print('Eigenvectors of Sun Mars Lagrange Point' , L2smvec)

print('Eigenvectors of Earth Moon Lagrange Point' , L2emovec)



#Task 4.1



XL2sm = np.hstack([L2sm[:2], [0, 0]])
XL2emo = np.hstack([L2emo[:2], [0, 0]])



# Identify real stable/unstable eigenvectors by eigenvalue, not column order.
def manifold_vector(eigenvalues, eigenvectors, stable):
    candidates = [i for i, value in enumerate(eigenvalues)
                  if abs(value.imag) < 1e-10
                  and (value.real < -1e-10 if stable else value.real > 1e-10)]
    if len(candidates) != 1:
        raise ValueError("Expected one real eigenvector for this manifold")
    vector = np.real(eigenvectors[:, candidates[0]])
    vector = vector/np.linalg.norm(vector)
    # Positive perturbations always start on the right of L2.
    return vector if vector[0] > 0 else -vector

stable_vector = manifold_vector(L2smval, L2smvec, stable=True)
unstable_vector = manifold_vector(L2smval, L2smvec, stable=False)
X0ps = XL2sm + eps*stable_vector    # stable right
X0ms = XL2sm - eps*stable_vector    # stable left
X0pu = XL2sm + eps*unstable_vector  # unstable right
X0mu = XL2sm - eps*unstable_vector  # unstable left

t_years=5

t_nd = t_years * t_year / T_sm


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
sol_sp = solve_ivp(cr3bp, t_backward, X0ps, args=(mu_sm,), rtol=1e-12, atol=1e-12, dense_output=True, max_step=0.05)
sol_sm = solve_ivp(cr3bp, t_backward, X0ms, args=(mu_sm,), rtol=1e-12, atol=1e-12, dense_output=True, max_step=0.05)

sol_up = solve_ivp(cr3bp, t_forward, X0pu, args=(mu_sm,), rtol=1e-12, atol=1e-12, dense_output=True, max_step=0.05)
sol_um = solve_ivp(cr3bp, t_forward, X0mu, args=(mu_sm,), rtol=1e-12, atol=1e-12, dense_output=True, max_step=0.05)


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
    if not sol.success:
        raise RuntimeError("Trajectory integration failed: " + sol.message)
    rho = np.linalg.norm(sol.y[:2].T - planet, axis=1)
    g = rho-r_nd
    for i in range(len(g)-1):
        if g[i] == 0:
            return sol.y[:, i].copy(), sol.t[i]
        if g[i]*g[i+1] <= 0:
            def crossing(t):
                return np.linalg.norm(sol.sol(t)[:2]-planet)-r_nd
            lo, hi = sorted((sol.t[i], sol.t[i+1]))
            t_int = brentq(crossing, lo, hi, xtol=1e-13)
            return sol.sol(t_int), t_int
    return None


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


# Classical transfers: inspect first crossings of both appropriate branches.
classical_earth = []
for branch_name, trajectory in [("stable right", sol_sp), ("stable left", sol_sm)]:
    crossing = first_intersection_state(trajectory, sun, r_end)
    if crossing is not None:
        state, time = crossing
        dv = capture_delta_v_sun(state, sun, L_sm, T_sm, gp_sun)
        classical_earth.append((dv, state, time, trajectory))
        print("Classical Earth crossing:", branch_name, "Delta-V [m/s]:", dv*1000)

classical_mars = []
for branch_name, trajectory in [("unstable right", sol_up), ("unstable left", sol_um)]:
    crossing = first_intersection_state(trajectory, mars, r_nd)
    if crossing is not None:
        state, time = crossing
        dv_pro, dv_ret = capture_delta_v(state, mars, mu_sm, L_sm, T_sm)
        classical_mars.append((min(dv_pro, dv_ret), state, time, trajectory))
        print("Classical Mars crossing:", branch_name,
              "prograde/retrograde Delta-V [m/s]:", dv_pro*1000, dv_ret*1000)

if not classical_earth:
    print("No classical stable-manifold crossing with Earth's orbit in five years.")
if classical_mars:
    dv_classical, X_cap_um, t_cap_um, capture_sol = min(classical_mars, key=lambda item: item[0])
    print("Capture state:", X_cap_um)
    print("Capture time [Earth years]:", t_cap_um*T_sm/t_year)
    print("Minimum classical Mars Delta-V [m/s]:", dv_classical*1000)
else:
    print("No classical unstable-manifold crossing with the Mars parking orbit in five years.")
if classical_earth and classical_mars:
    print("Minimum classical total Delta-V [m/s]:",
          (min(item[0] for item in classical_earth)+dv_classical)*1000)

plt.figure(figsize=(8, 6))

# Manifolds
plt.plot(sol_sp.y[0], sol_sp.y[1], 'b', linewidth=0.8,label ='stable right')
plt.plot(sol_sm.y[0], sol_sm.y[1], 'b--', linewidth=0.8,label ='stable left')
plt.plot(sol_up.y[0], sol_up.y[1], 'r', linewidth=0.8,label ='unstable right')
plt.plot(sol_um.y[0], sol_um.y[1], 'r--', linewidth=0.8,label ='unstable left')

plt.plot(orbit_x2, orbit_y2, 'g', linewidth=2, label='Earth Orbit')
# Mars-centered orbit
plt.plot(orbit_x, orbit_y, 'g', linewidth=2, label='Target orbit')
# Bodies
plt.scatter(sun[0], sun[1], color='orange', s=100, label='Sun')
plt.scatter(mars[0], mars[1], color='red', s=100, label='Mars')
plt.scatter(XL2sm[0], XL2sm[1], color='k', s=60, label='L2')

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
plt.plot(sol_up.y[0], sol_up.y[1], 'r', alpha=0.6,label ='unstable right')
plt.plot(sol_um.y[0], sol_um.y[1], 'r--', alpha=0.6,label ='unstable left')
# Mars-centered orbit
plt.plot(orbit_x, orbit_y, 'g', linewidth=2, label='Target orbit')

# Focus point: Mars
plt.scatter(mars[0], mars[1], color='red', s=100, label='Mars')

# L2 (still visible nearby structure)
plt.scatter(XL2sm[0], XL2sm[1], color='k', s=50, label='L2')

# ---- zoom limits ----
x_all = np.concatenate([
    sol_sp.y[0], sol_sm.y[0], sol_up.y[0], sol_um.y[0],
    orbit_x,
    [mars[0], XL2sm[0]]
])

y_all = np.concatenate([
    sol_sp.y[1], sol_sm.y[1], sol_up.y[1], sol_um.y[1],
    orbit_y,
    [mars[1], XL2sm[1]]
])

# Only keep points near Mars/L2 region to avoid zooming out too much
x_min_region = min(mars[0], XL2sm[0]) - 2*r_nd
x_max_region = max(mars[0], XL2sm[0]) + 2*r_nd
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
if classical_mars:
    # Trim the classical trajectory at the refined first intersection.
    mask_cap = capture_sol.t < t_cap_um
    x_seg = np.append(capture_sol.y[0, mask_cap], X_cap_um[0])
    y_seg = np.append(capture_sol.y[1, mask_cap], X_cap_um[1])

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

    # L2 point
    plt.scatter(
        XL2sm[0], XL2sm[1],
        color='black',
        s=70,
        label='Sun–Mars L2',
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

    # Zoom around L2, Mars orbit, and manifold segment
    x_all = np.concatenate([
        x_seg,
        orbit_x,
        [mars[0], XL2sm[0], X_cap_um[0]]
    ])

    y_all = np.concatenate([
        y_seg,
        orbit_y,
        [mars[1], XL2sm[1], X_cap_um[1]]
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
    plt.title("Close View: L2 to Mars-Orbit Capture Point")
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()



alphas = np.arange(-90, 90 + 2.5, 2.5)


beta = 0.03
viable_transfers = []

for alpha_deg in alphas:

    sol = solve_ivp(
        cr3bp_sail,
        t_backward,
        X0ps,
        args=(mu_sm, beta, alpha_deg),
        rtol=1e-12,
        atol=1e-12, dense_output=True, max_step=0.05
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
    key=lambda x: x["delta_v_ms"],
    default=None
)

if best_earth_case is not None:
    print("\nOPTIMAL EARTH-L2 TRANSFER")
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
        atol=1e-12, dense_output=True, max_step=0.05,
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
    key=lambda x: x["delta_v_ms"],
    default=None
)

if best_earth_case is None or best_mars_case is None:
    raise SystemExit("No complete solar-sail transfer found within five years for the assigned branches and cone angles.")

print("Optimal total Delta-V [m/s]:",
      best_earth_case["delta_v_ms"] + best_mars_case["delta_v_ms"])
print("\nOPTIMAL L2-MARS TRANSFER")
print("---------------------")
print("Cone angle [deg]:", best_mars_case["alpha_deg"])
print("Transfer time [yr]:", best_mars_case["time_years"])
print("Minimum Delta-V [m/s]:", best_mars_case["delta_v_ms"])

# Get optimal solutions and intersection states/times
best_leg1_sol = best_earth_case["solution"]   # backward: Earth orbit -> L2
best_leg2_sol = best_mars_case["solution"]    # forward: L2 -> Mars orbit

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


# Task 4.3 plots: dotted transfer legs, black orbits and insertion stars.
def transfer_axes(name):
    fig, ax = plt.subplots(num=name, figsize=(7, 6))
    ax.set_xlabel(r"$x$ [-]")
    ax.set_ylabel(r"$y$ [-]")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, color='0.7', linewidth=0.7)
    return fig, ax

def finish_transfer_plot(fig):
    # Save the three report figures beside this script as well as displaying them.
    from pathlib import Path
    plot_dir = Path(__file__).resolve().parent / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_dir / (fig.get_label() + ".png"), dpi=250)
    plt.show()

fig, ax = transfer_axes("optimal_transfer_system")
ax.plot(x_leg1, y_leg1, ':', color='tab:blue', lw=1.4, label='Transfer leg 1')
ax.plot(x_leg2, y_leg2, ':', color='tab:red', lw=1.4, label='Transfer leg 2')
ax.plot(orbit_x2, orbit_y2, 'k-', lw=1, label='Earth orbit')
ax.plot(orbit_x, orbit_y, 'k-', lw=1, label='MPO')
ax.scatter(*sun, color='tab:orange', s=25, label='Sun', zorder=5)
ax.scatter(*mars, color='firebrick', s=15, zorder=5)
ax.scatter(X_earth_int[0], X_earth_int[1], color='gold', marker='*', s=65,
           label='Transfer insertion', zorder=6)
ax.legend(loc='upper right', fontsize=9)
finish_transfer_plot(fig)

fig, ax = transfer_axes("optimal_transfer_mars")
ax.plot(x_leg1, y_leg1, ':', color='tab:blue', lw=1.4, label='Transfer leg 1')
ax.plot(x_leg2, y_leg2, ':', color='tab:red', lw=1.4, label='Transfer leg 2')
ax.plot(orbit_x, orbit_y, 'k-', lw=1, label='MPO')
ax.scatter(*mars, color='firebrick', s=25, label='Mars', zorder=5)
ax.scatter(XL2sm[0], XL2sm[1], color='tab:purple', marker='x', s=40,
           linewidths=0.9, label=r'$L_2$', zorder=5)
ax.scatter(X_mars_int[0], X_mars_int[1], color='gold', marker='*', s=70,
           label='MPO insertion', zorder=6)
# Keep a fixed local window: showing leg 1 must not zoom out to Earth.
ax.set_xlim(mars[0]-0.004, mars[0]+0.007)
ax.set_ylim(-0.004, 0.006)
ax.legend(loc='upper right', fontsize=9)
finish_transfer_plot(fig)

# Align the Earth and Mars curves to the original grid. NaNs preserve infeasible gaps.
earth_dv_dict = {case['alpha_deg']: case['delta_v_ms'] for case in viable_transfers}
mars_dv_dict = {case['alpha_deg']: case['delta_v_ms'] for case in viable_mars_transfers}
earth_alpha = alphas.copy()
mars_alpha = alphas.copy()
earth_dv = np.array([earth_dv_dict.get(a, np.nan) for a in alphas])
mars_dv = np.array([mars_dv_dict.get(a, np.nan) for a in alphas])
total_dv = earth_dv + mars_dv
common_alphas = alphas[np.isfinite(total_dv)]
fig, ax = plt.subplots(num="delta_v_cone_angle", figsize=(7, 5))
ax.plot(alphas, earth_dv, '-o', color='tab:blue', ms=3, lw=1, label=r'$\Delta V_E$')
ax.plot(alphas, mars_dv, '-o', color='tab:orange', ms=3, lw=1, label=r'$\Delta V_M$')
# The total-Delta-V curve is omitted: no sampled angle supports both legs.
ax.set_xlim(-90, 90)
ax.set_ylim(bottom=0)
ax.set_xlabel(r'$\alpha$ [deg]')
ax.set_ylabel(r'$\Delta V$ [m/s]')
ax.grid(True, color='0.7', linewidth=0.7)
ax.legend(loc='upper right')
finish_transfer_plot(fig)

#task 4,5



phi_0 = 122.35   # deg, Earth ahead of Mars

n_earth = 360.0              # deg/year
n_mars = 360.0 * t_year / (2 * np.pi * T_sm)
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


# Search forward from 1 January 2030
delta_phi_forward = (phi_req - phi_0) % 360

wait_years_forward = delta_phi_forward / n_rel

launch_date_after = start_date + timedelta(
    days=wait_years_forward * 365.256
)

print("Earliest matching launch date on or after 1 January 2030:", launch_date_after.date())

# Final results summary (computed values, not hard-coded results).
print("\n" + "=" * 76)
print("OPTIMAL SOLAR-SAIL EARTH-MARS TRANSFER: STUDENT 5207290".center(76))
print("=" * 76)
print(f"{'Transfer leg':<34}{'Angle [deg]':>14}{'Delta-V [m/s]':>16}{'Time [yr]':>12}")
print("-" * 76)
print(f"{'Leg 1: Earth orbit -> L2':<34}"
      f"{best_earth_case['alpha_deg']:>14.1f}"
      f"{best_earth_case['delta_v_ms']:>16.2f}"
      f"{best_earth_case['time_years']:>12.3f}")
print(f"{'Leg 2: L2 -> Mars parking orbit':<34}"
      f"{best_mars_case['alpha_deg']:>14.1f}"
      f"{best_mars_case['delta_v_ms']:>16.2f}"
      f"{best_mars_case['time_years']:>12.3f}")
print("-" * 76)
print(f"{'TOTAL':<34}{'':>14}"
      f"{best_earth_case['delta_v_ms'] + best_mars_case['delta_v_ms']:>16.2f}"
      f"{best_earth_case['time_years'] + best_mars_case['time_years']:>12.3f}")
print("=" * 76)
print("Times are in Earth years; total time is the sum of both transfer legs.")
