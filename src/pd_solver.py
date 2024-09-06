from pd_input import pd_input
from mpi4py import MPI
import numpy as np
import logging,os
from pandas import DataFrame
out_print = lambda text: os.system(f"ECHO {text}")

class pd_solver:
    logging.basicConfig(
        filename='run.log',
        filemode = 'w',
        format='%(asctime)s, %(name)s: %(levelname)s - %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')

    def __init__(self):
        self.logger = logging.getLogger('SOLVER')
        self.run_vec()

    def run_vec(self):
        self.input = pd_input(self)
        self.logger.info('Inputs are gathered.')
        self.mpi_setup_vec()
        self.logger.info('MPI setup is finished.')
        self.initialize_vec()
        self.logger.info('Neighborhood arrays are initialized.')
        self.logger.info('Iteration begin.')
        self.iterate_vec()

    def mpi_setup_vec(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_points,self.num_dofs = self.coord.shape

        AVERAGE_FAM_SIZE = (self.from_to[:,1]-self.from_to[:,0]).mean()
        AVERAGE_INTERACTION_PER_PROCESS = int(len(self.neighbors)/self.size)
        
        CURRENT_RANK=0
        INTERACTION_COUNTER=0
        FAMILY_COUNTER=0
        self.family_list = []

        for FAMILY,(f,t) in enumerate(self.from_to):
            family_size = t-f
            cond1 = INTERACTION_COUNTER>=AVERAGE_INTERACTION_PER_PROCESS-AVERAGE_FAM_SIZE
            cond2 = INTERACTION_COUNTER<=AVERAGE_INTERACTION_PER_PROCESS+AVERAGE_FAM_SIZE

            if not (cond1 and cond2):
                INTERACTION_COUNTER+=family_size
            else:
                if CURRENT_RANK==self.rank:
                    if not self.rank==(self.size-1):
                        self.pd_interactions = np.array([self.from_to[FAMILY_COUNTER,0],self.from_to[FAMILY,0]])
                        self.family_list = list(range(FAMILY_COUNTER,FAMILY))
                    else:
                        self.pd_interactions = np.array([self.from_to[FAMILY_COUNTER,0],self.from_to[-1,1]])
                        self.family_list = list(range(FAMILY_COUNTER,len(self.from_to)))
                    out_print(f'Process {self.rank} has pd interactions: {self.pd_interactions}, {self.pd_interactions[1]-self.pd_interactions[0]}')
                    out_print(f'Process {self.rank} has {len(self.family_list)} families and {INTERACTION_COUNTER} interactions. Family list starts from {self.family_list[0]} to {self.family_list[-1]}')
                    self.pd_points=self.family_list
                    self.reduction_indices = [int(self.from_to[curr][0]) - int(self.pd_interactions[0])  for curr in self.family_list]
                    self.reduction_indices.append(int(self.from_to[self.family_list[-1]][-1]- int(self.pd_interactions[0])))
                INTERACTION_COUNTER = family_size
                CURRENT_RANK+=1
                FAMILY_COUNTER=FAMILY

    def iterate_vec(self):
        self.iter = 0
        self.cn = np.zeros(shape=self.config['max_iter'])
        while self.iter<=self.config['max_iter']:
            self.apply_bc()
            self.iterate_initializer_vec()
            if self.config['failure']==True:
                self.compute_damage_vec()
            self.compute_dilation_vec()
            self.compute_force_vec()
            self.ADR()
            if self.rank==0 and self.iter % 200 == 199:
                self.output()
            self.update_state()

    def initialize_vec(self):
        self.delta = np.float64(self.config['delta'])

        self.disp       = np.zeros_like(self.coord)
        self.velhalf    = np.zeros_like(self.coord)
        self.velhalfold = np.zeros_like(self.coord)
        self.force      = np.zeros_like(self.coord)
        self.forceold   = np.zeros_like(self.coord)
        self.bforce     = np.zeros_like(self.coord)
        self.dilat      = np.zeros_like(self.coord[:,0])

        f = self.pd_interactions[0]
        t = self.pd_interactions[1]

        self.local_interactions=[0,t-f]

        self.xi         = self.coord[self.neighbors[f:t,1]] - self.coord[self.neighbors[f:t,0]]
        self.xi_norm    = np.linalg.norm(self.xi,axis=1)
        
        self.volume_curr    = self.volumes[self.neighbors[f:t,0]]
        self.volume_other   = self.volumes[self.neighbors[f:t,1]]
        self.volume_other_  = np.append(self.volume_other,0)
        

        self.eta        = np.zeros_like(self.xi)
        self.y          = np.zeros_like(self.xi)
        self.eta_norm   = np.zeros_like(self.xi_norm)
        self.y_norm     = np.zeros_like(self.xi_norm)

        self.stretch = np.zeros_like(self.xi_norm)
        self.lambdas = np.zeros_like(self.xi_norm)
        self.dilatkj = np.append(np.zeros_like(self.xi_norm),0)
        self.forcekj = np.append(np.zeros_like(self.xi),[[0,0]],axis=0)
        
        self.dilat_curr = np.zeros_like(self.volume_curr)
        self.dilat_other = np.zeros_like(self.volume_curr)

        shape           = (self.size,self.num_points)
        self.dummy_scal = np.zeros(shape=shape,dtype=np.float64)

        shape           = (self.size,self.num_points,self.num_dofs)
        self.dummy_vec  = np.zeros(shape=shape,dtype=np.float64)

        if self.config['failure']==True:
            self.damage_index = np.zeros_like(self.coord[:,0])
            self.damage_index_=np.zeros_like(self.damage_index)
            self.mus = np.ones_like(self.xi_norm)
            self.damage_index_kj_= np.append(np.zeros_like(self.xi_norm),0)


    def compute_damage_vec(self):
        #Compute Damage Index
        [f,t]               = self.pd_interactions
        self.mus[:] = ((self.stretch<self.sc[f:t])&(self.mus==1)).astype(int)
        self.a[f:t]*=self.mus
        self.b[f:t]*=self.mus
        self.d[f:t]*=self.mus
        self.damage_index_kj_[:-1] = self.mus*self.volume_other
        self.damage_index_[:]=0.0

        self.damage_index_[self.family_list[0]:self.family_list[-1]+1] = 1.0 - np.add.reduceat(self.damage_index_kj_,self.reduction_indices,axis=0)[:-1] / np.add.reduceat(self.volume_other_,self.reduction_indices,axis=0)[:-1]
        #Gather Damage Index
        [f,t]           =  self.pd_interactions
        # shape           = (self.size,self.num_points)
        # dummy           = np.zeros(shape=shape,dtype=np.float64)
        self.dummy_scal[:] = 0.0
        self.comm.Allgather( self.damage_index_.copy() , self.dummy_scal )
        self.damage_index[:] = self.dummy_scal.sum(axis=0)
        #-----------------

    def apply_bc(self):
        # # Ramped
        t0 = self.config['t0']
        max_iter = self.config['max_iter']
        # Apply BCs
        if self.config.__contains__('bc_app'):
            if self.config['bc_app'].upper()=='INCREMENTAL':
                step = 4000
                if self.iter % step ==0:
                    fac = self.iter
                else:
                    fac = self.iter-self.iter%step
        else:
            fac = self.iter
        if self.applied['bforce']['index'].any():
            if self.iter<=int(max_iter*t0):
                self.bforce[self.applied['bforce']['index'],:] = self.applied['bforce']['values']/(max_iter*t0)*fac
            elif self.iter>int(max_iter*t0):
                self.bforce[self.applied['bforce']['index'],:] = self.applied['bforce']['values']
        elif self.applied['disp']['index'].any():
            applied_disp = self.applied['disp']
            if self.iter<=int(max_iter*t0):
                self.disp[self.applied['disp']['index'],self.applied['disp']['dof']] = self.applied['disp']['appliedd']/(max_iter*t0)*fac
            elif self.iter>int(max_iter*t0):
                self.disp[self.applied['disp']['index'],self.applied['disp']['dof']] = self.applied['disp']['appliedd']

    def compute_dilation_vec(self):
        #Compute Dilation
        [f,t]               = self.pd_interactions
        self.dilatkj[:-1]     = self.stretch  * self.lambdas *self.volume_other * self.d[f:t] *self.delta
        self.dilat[self.family_list[0]:self.family_list[-1]+1]  = np.add.reduceat(self.dilatkj,self.reduction_indices)[:-1]
        #Gather Dilat
        [f,t]           =  self.pd_interactions
        # shape           = (self.size,self.num_points)
        # dummy           = np.zeros(shape=shape,dtype=np.float64)
        self.dummy_scal[:] = 0.0
        self.comm.Allgather( self.dilat.copy() , self.dummy_scal )
        self.dilat[:]   = self.dummy_scal.sum(axis=0)
        self.dilat_curr[:]=  self.dilat[self.neighbors[f:t,0]]
        self.dilat_other[:] =  self.dilat[self.neighbors[f:t,1]]
        #-----------------

    def compute_force_vec(self):
        [f,t]           =  self.pd_interactions
        A               = 2 * self.a[f:t] *  self.d[f:t] * self.delta * self.lambdas / self.xi_norm * self.dilat_curr    + 2 *  self.b[f:t] * self.delta * self.stretch
        B               = 2 * self.a[f:t] *  self.d[f:t] * self.delta * self.lambdas / self.xi_norm * self.dilat_other   + 2 *  self.b[f:t] * self.delta * self.stretch
        tij             =   self.y * ( A / self.y_norm).reshape(self.y_norm.shape[0],1)
        tji             =   -self.y * ( B / self.y_norm).reshape(self.y_norm.shape[0],1)
        self.forcekj[:-1,:] = (tij-tji) * (self.volume_other ).reshape(len(self.volume_other),1)
        self.force[self.family_list[0]:self.family_list[-1]+1,:] = np.add.reduceat(self.forcekj,self.reduction_indices,axis=0)[:-1]
        
        #Gather Force
        [f,t]            =  self.pd_interactions
        # shape            = (self.size,self.num_points,self.num_dofs)
        # dummy            = np.zeros(shape=shape,dtype=np.float64)
        self.dummy_vec[:] = 0.0
        self.comm.Allgather( self.force.copy() , self.dummy_vec )
        self.force[:,:]  = self.dummy_vec.sum(axis=0)
        #-----------------

    def ADR(self):
        dt = self.config['dt']
        f = self.family_list[0]
        t = self.family_list[-1]+1
        #ADR
        if self.iter==0:
            self.velhalf[f:t]= (self.force[f:t]+self.bforce[f:t]) * dt / self.dii[f:t] / 2
        else:
            cn = 0.0
            a1 = 0.0
            b1 = 0.0

            non_zero_velhalf_ind = np.where(self.velhalfold[f:t].ravel()!=0.0)[0]

            K_ii = - (self.force[f:t] - self.forceold[f:t]) /self.dii[f:t] / dt
            K_ii = K_ii.ravel()[non_zero_velhalf_ind]
            K_ii /= self.velhalfold[f:t].ravel()[non_zero_velhalf_ind]
            
            
            a1 = (np.pow(self.disp[f:t].ravel()[non_zero_velhalf_ind],2.0) * K_ii).sum(axis=0)
            b1 = np.pow(self.disp[f:t].ravel(),2.0).sum()
            a1 = self.comm.allreduce(a1,op=MPI.SUM)
            b1 = self.comm.allreduce(b1,op=MPI.SUM)

            if not b1==0.0:
                if a1/b1>0.0:
                    cn = 2 * np.sqrt(a1/b1)
                else:
                    cn = 0.0
            else:
                cn = 0.0
            if cn>2.0:
                cn = 1.9
            a1 = 2 - cn * dt
            a2 = 2 * dt
            b1 = 2 + cn * dt
            self.velhalf[f:t,:]=(a1 * self.velhalfold[f:t] + a2 * (self.force[f:t]+self.bforce[f:t]) / self.dii[f:t])/b1
        #-----------------
        #Gather Velhalf
        # shape = (self.size,self.num_points,self.num_dofs)
        # dummy = np.zeros(shape=shape,dtype=np.float64)
        self.dummy_vec[:] = 0.0
        self.comm.Allgather(self.velhalf.copy(),self.dummy_vec)
        self.velhalf[:,:] = self.dummy_vec.sum(axis=0)
        #-----------------

    def update_state(self):
            self.disp += self.velhalf * self.config['dt']
            self.velhalfold[:] = self.velhalf
            self.forceold[:] = self.force
            self.iter+=1


    def iterate_initializer_vec(self):
        self.dilat[:]      = 0.0
        self.force[:]      = 0.0
        self.velhalf[:]    = 0.0

        f = self.pd_interactions[0]
        t = self.pd_interactions[1]
        

        self.eta[:]         = self.disp[self.neighbors[f:t,1]] - self.disp[self.neighbors[f:t,0]]
        self.y[:]           = self.xi + self.eta
        self.eta_norm[:]    = np.linalg.norm(self.eta,axis=1)
        self.y_norm[:]      = np.linalg.norm(self.y,axis=1)
        self.stretch[:]     = (self.y_norm) / self.xi_norm - 1.0
        self.lambdas[:]     = (self.y * self.xi).sum(axis=1) / self.y_norm / self.xi_norm
        
    def output(self):
        if not hasattr(self,'output_df'):
            self.df = DataFrame()
        self.df = DataFrame()
        self.df[['coordx','coordy']] = self.coord
        self.df[['dispx','dispy']] = self.disp
        self.df[['deformedx','deformedy']] = self.coord+self.disp
        self.df[['forcex','forcey']] = self.force
        self.df[['bforcex','bforcey']] = self.bforce
        self.df[['forceoldx','forceoldy']] = self.forceold
        self.df[['velhalfx','velhalfy']] = self.velhalf
        self.df[['velhalfoldx','velhalfoldy']] = self.velhalfold
        self.df[['diix','diiy']] = self.dii
        self.df['dilat'] = self.dilat
        if self.config['failure']:
            self.df['damage_index'] = self.damage_index
        fname  = f'output_{self.iter}.csv'
        self.df.to_csv(fname,index=False)

prog=pd_solver()