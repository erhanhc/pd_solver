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
        self.run()

    def mpi_setup(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_points,self.num_dofs = self.coord.shape
        from_ = int((self.num_points/self.size)*self.rank)
        to_   = int((self.num_points/self.size)*(self.rank+1))
        self.pd_points = list(
                        range(
                            from_,
                            to_ if to_<=self.num_points else self.num_points
                            )
                        )

    def run(self):
        self.input = pd_input(self)
        self.logger.info('Inputs are gathered.')
        self.mpi_setup()
        self.logger.info('MPI setup is finished.')
        self.initialize()
        self.logger.info('Empty arrays are initialized.')
        self.logger.info('Iteration begin.')
        self.iterate()

    def iterate(self):
        self.iter = 0
        self.cn = []
        while self.iter<=self.config['max_iter']:
            self.apply_bc()
            if self.rank==0 and self.iter % 100 == 0:
                self.output()
            if self.config['failure']==True:
                self.compute_damage()
            self.iterate_initializer()
            self.compute_dilation()
            self.compute_force()
            self.ADR()
            self.update_state()

    def initialize(self):
        self.disp = np.zeros_like(self.coord)
        self.velhalf = np.zeros_like(self.coord)
        self.velhalfold = np.zeros_like(self.coord)
        self.force = np.zeros_like(self.coord)
        self.forceold = np.zeros_like(self.coord)
        self.bforce = np.zeros_like(self.coord)
        self.dilat = np.zeros_like(self.coord[:,0])
        if self.config['failure']==True:
            self.damage_index = np.zeros_like(self.coord[:,0])
            self.mus = np.ones_like(self.sc)

    def compute_damage(self):
        self.damage_index_=np.zeros_like(self.damage_index)
        for curr in self.pd_points:
            f,t = self.from_to[curr]
            others = self.neighbors[f:t,1]

            sc = self.sc[f:t]
            mu = self.mus[f:t].copy()

            x = self.coord[others] - self.coord[curr]
            x_norm = np.linalg.norm(x,axis=1)

            y = x + self.disp[others] - self.disp[curr]
            y_norm = np.linalg.norm(y,axis=1)

            stretchs = y_norm - x_norm
            stretchs /= x_norm

            self.mus[f:t] = ((stretchs<sc)&(mu==1)).astype(int)

            self.a[f:t]*=self.mus[f:t]
            self.b[f:t]*=self.mus[f:t]
            self.d[f:t]*=self.mus[f:t]

            self.damage_index_[curr] = 1 - ((self.mus[f:t]*self.volumes[others]).sum())/(self.volumes[others].sum())
        #Gather Damage Indices
        shape = (self.size,self.num_points)
        dummy = np.zeros(shape=shape,dtype=np.float64)
        self.comm.Allgather(self.damage_index_.copy(),dummy)
        self.damage_index = dummy.sum(axis=0).copy()
        #-----------------

    def apply_bc(self):
        t0 = self.config['t0']
        max_iter = self.config['max_iter']
        # Apply BCs
        if self.applied['bforce']['index'].any():
            if self.iter<=int(max_iter*t0):
                self.bforce[self.applied['bforce']['index'],:] = self.applied['bforce']['values']/(max_iter*t0)*self.iter
            elif self.iter>int(max_iter*t0):
                self.bforce[self.applied['bforce']['index'],:] = self.applied['bforce']['values']
        elif self.applied['disp']['index'].any():
            applied_disp = self.applied['disp']
            for i,d,v in zip(applied_disp['index'],applied_disp['dof'],applied_disp['appliedd']):
                if self.iter<=int(max_iter*t0):
                    self.disp[i,d] = v/(max_iter*t0)*self.iter
                elif self.iter>int(max_iter*t0):
                    self.disp[i,d] = v
        #-----------------

    def compute_dilation(self):
        delta = self.config['delta']
        #Compute Dilation
        for curr in self.pd_points:
            f,t = self.from_to[curr]
            others = self.neighbors[f:t,1]

            d = self.d[f:t]

            x = self.coord[others] - self.coord[curr]
            x_norm = np.linalg.norm(x,axis=1)

            y = x + self.disp[others] - self.disp[curr]
            y_norm = np.linalg.norm(y,axis=1)
            
            stretchs = y_norm - x_norm
            stretchs /= x_norm
            
            Lambdas = 1 / y_norm / x_norm
            Lambdas *= (y*x).sum(axis=1)

            self.dilat[curr] = (stretchs * Lambdas *self.volumes[others] * d * delta).sum(axis=0)
        #-----------------
        #Gather Dilat
        shape = (self.size,self.num_points)
        dummy = np.zeros(shape=shape,dtype=np.float64)
        self.comm.Allgather(self.dilat.copy(),dummy)
        self.dilat = dummy.sum(axis=0).copy()
        #-----------------

    def compute_force(self):
        delta = self.config['delta']
        #Compute Force
        for curr in self.pd_points:
            f,t = self.from_to[curr]
            others = self.neighbors[f:t,1]
            a = self.a[f:t]
            b = self.b[f:t]
            d = self.d[f:t]

            x = self.coord[others] - self.coord[curr]
            x_norm = np.linalg.norm(x,axis=1)
            
            y = x + self.disp[others] - self.disp[curr]
            y_norm = np.linalg.norm(y,axis=1)
            
            stretchs = y_norm - x_norm
            stretchs /= x_norm
            
            Lambdas = 1 / y_norm / x_norm
            Lambdas *= (y*x).sum(axis=1)
            
            A = 2 * a *  d * delta * Lambdas / x_norm * self.dilat[curr]    + 2 *  b * delta * stretchs
            B = 2 * a *  d * delta * Lambdas / x_norm * self.dilat[others]  + 2  * b * delta * stretchs
            
            tij =   y / y_norm.reshape(len(others),1) * A.reshape(len(others),1)
            tji = - y / y_norm.reshape(len(others),1) * B.reshape(len(others),1)

            self.force[curr,:] = ((tij-tji) * (self.volumes[others]).reshape(len(others),1)).sum(axis=0)
            #-----------------
        #Gather Force
        shape = (self.size,self.num_points,self.num_dofs)
        dummy = np.zeros(shape=shape,dtype=np.float64)
        self.comm.Allgather(self.force.copy(),dummy)
        self.force = dummy.sum(axis=0).copy()
        #-----------------

    def ADR(self):
        dt = self.config['dt']
        #ADR
        if self.iter==0:
            self.velhalf[self.pd_points]= (self.force[self.pd_points]+self.bforce[self.pd_points]) * dt / self.dii[self.pd_points] / 2
        else:
            cn = 0.0
            a1 = 0.0
            b1 = 0.0

            non_zero_velhalf_ind = np.where(self.velhalfold[self.pd_points].flatten()!=0.0)[0]
            K_ii = - (self.force[self.pd_points] - self.forceold[self.pd_points]) /self.dii[self.pd_points] / dt
            K_ii = K_ii.flatten()[non_zero_velhalf_ind]
            K_ii /= self.velhalfold[self.pd_points].flatten()[non_zero_velhalf_ind]
            
            
            a1 = ((self.disp[self.pd_points].flatten()[non_zero_velhalf_ind])**2 *  K_ii).sum(axis=0)
            b1 = self.disp[self.pd_points].flatten().dot(self.disp[self.pd_points].flatten())
            
            a1 = self.comm.allreduce(a1,op=MPI.SUM)
            b1 = self.comm.allreduce(b1,op=MPI.SUM)

            if not b1==0.0:
                if a1/b1>0.0:
                    cn = 2 * (a1/b1)**0.5
                else:
                    cn = 0.0
            else:
                cn = 0.0
                
            if cn>2.0:
                cn = 1.9
            self.cn.append([self.iter,cn])
            a1 = 2 - cn * dt
            a2 = 2 * dt
            b1 = 2 + cn * dt
            self.velhalf[self.pd_points,:]=(a1 * self.velhalfold[self.pd_points] + a2 * (self.force[self.pd_points]+self.bforce[self.pd_points]) / self.dii[self.pd_points])/b1
        #-----------------
        #Gather Velhalf
        shape = (self.size,self.num_points,self.num_dofs)
        dummy = np.zeros(shape=shape,dtype=np.float64)
        self.comm.Allgather(self.velhalf.copy(),dummy)
        self.velhalf = dummy.sum(axis=0).copy()
        #-----------------

    def update_state(self):
            self.disp += self.velhalf * self.config['dt']
            self.velhalfold = self.velhalf.copy()
            self.forceold = self.force.copy()
            self.iter+=1

    def iterate_initializer(self):
        self.dilat = np.zeros_like(self.coord[:,0])
        self.force = np.zeros_like(self.coord)
        self.velhalf = np.zeros_like(self.coord)

    def output(self):
        df = DataFrame()
        df[['coordx','coordy']] = self.coord
        df[['dispx','dispy']] = self.disp
        df[['deformedx','deformedy']] = self.coord+self.disp
        df[['forcex','forcey']] = self.force
        df[['bforcex','bforcey']] = self.bforce
        df[['forceoldx','forceoldy']] = self.forceold
        df[['velhalfx','velhalfy']] = self.velhalf
        df[['velhalfoldx','velhalfoldy']] = self.velhalfold
        df[['diix','diiy']] = self.dii
        df['dilat'] = self.dilat
        if self.config['failure']:
            df['damage_index'] = self.damage_index
        fname  = f'output_{self.iter}.csv'
        df.to_csv(fname,index=False)
        df = DataFrame(self.cn,columns=['iter','cn'])
        df.to_csv('cn.csv')

prog=pd_solver()