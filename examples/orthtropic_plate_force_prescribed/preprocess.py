import numpy as np
from scipy.spatial import KDTree
from pandas import DataFrame

def Plate(W,L,thickness,dx):
    V = L * W * thickness #mm3
    ndivx = int(L/dx)
    ndivy = int(W/dx)
    coord = np.array(
        [
        [-1/2 * L + dx / 2 + (i-1)*dx , -1/2 * W + dx/2 + (j-1)*dx] 
        for i in range(1,ndivx+1) 
        for j in range(1,ndivy+1)
        ]
        )
    volumes = np.array([V/coord.shape[0] for i in range(len(coord))])
    delta = dx*3.015
    return coord,volumes,delta

def neigh_search(coord,delta):
    tree = KDTree(coord)
    neighbors = [tree.query_ball_point(coord[curr],r = delta) for curr in range(len(coord))]
    neighbors = [ np.array([other for other in neig_list if not curr==other])for curr,neig_list in enumerate(neighbors)]
    i=0
    data1 = []
    data2 = []
    for curr,neig_list in enumerate(neighbors):
        from_ = i
        neig_list.sort()
        for other in neig_list:
            data1.append([curr,other])
            i+=1
        to=i
        data2.append([from_,to])
    neighbors = np.array(data1,dtype=int)
    from_to = np.array(data2,dtype=int)
    return neighbors,from_to

def pd_material_prop_iso(E,nu,delta,thickness):
    Mu = E/2/(1+nu)
    Kappa = E/2/(1-nu)
    a = 1 / 2 * (Kappa - 2 * Mu)#MPa = N/mm2
    b = 6 * Mu / np.pi /thickness / delta**4 #MPa/mm/mm4 = N/mm7
    d = 2 / np.pi / thickness / delta **3# 1/mm4
    return a,b,d

def pd_material_prop_ortho(E11,E22,G12,nu12,thickness,delta,dx):
    nu21 = nu12 * E22 / E11
    Q11 = E11/(1-nu12*nu21)
    Q22 = E22/(1-nu12*nu21)
    Q12 = nu12 * E22 / (1-nu12*nu21)
    Q66 = G12
    H = (np.pi*thickness*delta**2)
    n = int(delta/dx)

    a = 1/2 * ( Q12 - Q66) #N/mm2
    bft = 6*Q66 / np.pi / thickness / delta**4 #N/mm7
    bf = (Q11 - Q12 - 2*Q66)/(2*delta*H*delta*(n+1)/int(np.pi*n**2+1))#N/mm7
    bt = (Q22 - Q12 - 2*Q66)/(2*delta*H*delta*(n+1)/int(np.pi*n**2+1))#N/mm7
    d = 2 / np.pi / thickness / delta **3# 1/mm4
    return a,bf,bt,bft,d

def compute_dii(coord,from_to,neighbors,volumes,a,b,d,delta,mass_vec_safety_factor=5.0,dt=1.0):
    dii = np.zeros_like(coord)
    for curr in range(len(coord)):
        f,t = from_to[curr]
        others = neighbors[f:t,1]

        ksis = coord[others] - coord[curr]
        ksi_norm = np.linalg.norm(ksis,axis=1)
        
        a1s = 4 * delta / ksi_norm**2
        b1s = 0.5 * a[f:t] * d[f:t]**2 * delta / ksi_norm * (volumes[curr]+volumes[others]) + b[f:t]
        sumijs = a1s * b1s
        sumijs = abs(ksis) * (sumijs*volumes[others]).reshape(len(others),1)
        dii[curr,:] = 1/4*dt**2*sumijs.sum(axis=0)*mass_vec_safety_factor
    return dii

def apply_uniaxial_bforce(coord,dx,appres):
    bforce = np.zeros_like(coord)
    left_bound = np.where(coord[:,0]<=min(coord[:,0]))[0]
    right_bound = np.where(coord[:,0]>=max(coord[:,0]))[0]

    bforce[left_bound,0] = -1 * appres / dx
    bforce[right_bound,0] = +1 * appres / dx
    df = DataFrame(bforce,columns=['appliedb1','appliedb2'])
    df = df[(df['appliedb1']!=0.0)|(df['appliedb2']!=0.0)]
    return df

def orthotropy_calculations(ply_angle,alignment_tolerance,coord,neighbors,from_to,a,d,bf,bt,bft):
    global_fiber_direction = np.array([np.cos(np.deg2rad(ply_angle)),np.sin(np.deg2rad(ply_angle))])
    global_fiber_direction = global_fiber_direction/np.linalg.norm(global_fiber_direction) 
    a = [a]*len(neighbors)
    d = [d]*len(neighbors)
    b = [] 
    for curr in range(len(coord)):
        f,t = from_to[curr]
        others = neighbors[f:t,1]
        ksis = coord[others] - coord[curr]
        ksi_norm = np.linalg.norm(ksis,axis=1)
        ksis_dir = ksis/ksi_norm.reshape((ksi_norm.shape[0],1))
        mu_f = (abs((global_fiber_direction*ksis_dir).sum(axis=1))>=np.cos(np.deg2rad(alignment_tolerance))).astype(np.float64)
        mu_t = (abs((global_fiber_direction*ksis_dir).sum(axis=1))<=np.sin(np.deg2rad(alignment_tolerance))).astype(np.float64)

        bs = mu_f*bf + bft + mu_t*bt
        [b.append(b_) for b_ in bs]
    return np.array(a),np.array(b),np.array(d)

def main(W,L,thickness,dx,E11,E22,G12,nu12,appres):
    coord,volumes,delta = Plate(W,L,thickness,dx)
    neighbors,from_to = neigh_search(coord,delta)
    a,bf,bt,bft,d = pd_material_prop_ortho(E11,E22,G12,nu12,thickness,delta,dx)
    a,b,d = orthotropy_calculations(ply_angle,alignment_tolerance,coord,neighbors,from_to,a,d,bf,bt,bft)

    dii = compute_dii(coord,from_to,neighbors,volumes,a,b,d,delta,mass_vec_safety_factor=5.0,dt=1.0)

    df1 = DataFrame(coord,columns = ['coord1','coord2'])
    df1['volume'] = volumes
    df1[['dii1','dii2']] = dii
    df1[['from','to']] = from_to
    df1.to_csv('body.csv')

    df2 = DataFrame(neighbors,columns = ['curr','other'])
    df2['a'] = a
    df2['b'] = b
    df2['d'] = d
    df2.to_csv('neigh.csv')

    df3 = apply_uniaxial_bforce(coord,dx,appres)
    df3.to_csv('appliedb.csv')

    df4 = DataFrame([],columns=['dof','appliedd'])
    df4.to_csv('appliedd.csv')
    
    with open('config.yaml','w') as out:
        line =f"""max_iter: 1000
failure: False
t0 : 0.7
delta : {delta}
dt : 1.0"""
        out.writelines(line)

if __name__=="__main__":
    L=100.#mm
    W = 100.#mm
    thickness = 0.18#mm
    E11 = 159.96 * 10**9#Pa
    E11/= 1000**2#MPa
    E22 = 8.96 * 10**9#Pa
    E22 /= 1000**2#MPa
    G12 = 3.0054 * 10**9#Pa
    G12 /= 1000**2#MPa
    nu12 = 1/3
    dx = L/100.
    appres = E11/1e3#MPa
    ply_angle=0.0
    alignment_tolerance = 15.0
    main(W,L,thickness,dx,E11,E22,G12,nu12,appres)