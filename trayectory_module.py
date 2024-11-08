import numpy as np 
from dataclasses import dataclass
from typing import Callable
from engine_module import *

@dataclass
class TrayectoryPhysicalConstants:
    """Physical constants that affect the trayectory"""
    g0: float       # Intensity of gravity field on the surface
    Ra: float       # Air constant
    Re: float       # Earth Radious

    def __post_init__(self):
        if any(value <= 0 for value in [self.g0, self.Ra, self.Re]): 
            raise ValueError("All values must be positive numbers.")

@dataclass
class TrayectoryInitialConditions:
    """Initial conditions for solving the trayectory SDE """
    t0: float       # Starting simulation time
    h0: float       # Initial heigth 
    v0: float       # Initial velocity 

    def __post_init__(self):
        if self.h0 < 0:
            raise ValueError("Initial heigth must be greater or equal to cero")
@dataclass
class TrayectoryGeometryParameters:
    """Geometry parameters of the rocket that affect the trayectory."""
    M0: float       # Inital mass of the body 
    S: float        # Transverse surface area of the body

    def __post_init__(self):
        if any(value <= 0 for value in [self.M0, self.S]): 
            raise ValueError("All values must be positive numbers.")

@dataclass
class TrayectorySimulationConfiguration:
    dt: float                           # time step 
    solver: str                         # RK4 or Euler
    def __post_init__(self):
        if self.dt <= 0:
            raise ValueError("Time step must be a poistive non zero value")
class trayectory:
    """ Calculates the trayectory of body in the Eath's atmosphere considering variable gravity intensity, 
    ISA Atmospehric properities with the option to include variable thrust, drag and mass.

    """
    # ISA atmospherica values
    atm = np.array([
        [0, 0, -6.5e-3, 288.15, 101325, 1.224],         # Troposfera <11000
        [11000, 11019, 0, 266.65, 22632, 0.3639],       # Tropopausa <20000
        [20000, 20063, 1e-3, 216.65, 5474.9, 0.088],    # Estratosfera <32000
        [32000, 32162, 2.8e-3, 228.65, 868.02, 0.0132], # Estratosfera 2 <47000
        [47000, 47350, 0, 270.65, 110.91, 1.224],       # Estratopausa <51000
        [51000, 51413, -2.8e-3, 270.65, 66.939, 0],     # Mesosfera <71000
        [71000, 71802, -2e-3, 214.65, 3.9564, 0],       # Mesosfera 2 <84532
        [84852, 86000, 0, 186.87, 0.3734, 0]            # Mesopausa < Termosfera
        ])
    
    def __init__(self,
                trayectory_physical_constants: TrayectoryPhysicalConstants, 
                trayectory_initial_conditions: TrayectoryInitialConditions,
                trayectory_geometry_parameters: TrayectoryGeometryParameters,
                trayectory_simulation_configuration: TrayectorySimulationConfiguration
    ):
        # Dictionary for available solvers
        self.solvers = {
            'Euler': self.Euler_step,
            'RK4': self.RK4_step
        }

        # Storing all class attibutes as attibutes in the rocket class
        self.__dict__.update(trayectory_physical_constants.__dict__)
        self.__dict__.update(trayectory_initial_conditions.__dict__)
        self.__dict__.update(trayectory_geometry_parameters.__dict__)
        self.__dict__.update(trayectory_simulation_configuration.__dict__)

        # Initialiting conditions and constants 
        self.init_conditions()
        self.init_variables()

    def init_conditions(self):
        """Inits the conditions for the SDE and the lists to store the values"""
        self.h_t = [self.h0]
        self.v_t = [self.v0]
    
    def init_variables(self):
        """Inits lists to store properties of the engine"""
        self.t = [self.t0]
        self.Pa_t = [self.atm_properties(self.h0)[0]]

    def atm_properties(self, h:float):
        """
        Provides the atmospheric properties: temperature, pressure and density for a given altitutude

        Args:
            h (float): heigth over the surface 

        Returns:
            float, float, float: Pressure, density and temperature
        """
        z = self.Re * h / (self.Re + h)
        k = np.where(self.atm[:, 0] <= z)[0][-1]
        if self.atm[k, 2] == 0:
            Pa = self.atm[k, 4] * np.exp(-self.g0 / (self.Ra * self.atm[k, 3]) * (z - self.atm[k, 0]))
            rho_ext = self.atm[k, 5] * np.exp(-self.g0 / (self.Ra * self.atm[k, 3]) * (z - self.atm[k, 0]))
            T_ext = self.atm[k, 3]
        else:
            T_ext = self.atm[k, 3] + self.atm[k, 2] * (z - self.atm[k, 0])
            Pa = self.atm[k, 4] * (T_ext / self.atm[k, 3]) ** (-self.g0 / (self.atm[k, 2] * self.Ra))
            rho_ext = self.atm[k, 5] * (T_ext / self.atm[k, 3]) ** (-self.g0 / (self.atm[k, 2] * self.Ra))
        
        return Pa, rho_ext, T_ext
    
    def diff_eq_system(self, h:float, v:float, E:float, M:float, Cd:float):
        """System of differential equations (SDE) for the trayectory. Values to solve are height and velocity.

        Args:
            h (float): heigth in the current instant
            v (float): velocity in the current instant
            E (float): thrust in the current instant
            M (float): mass in the current instant
            Cd (float): drag coefficient in the current time

        Returns:
            _type_: _description_
        """
        _, rho_ext, _ = self.atm_properties(h)   # calculating current atm density 

        a = (E - M * self.g0 - np.sign(v) * 0.5 * rho_ext * self.S * Cd * v ** 2) / M  
        
        dh_dt = v               # derivative of the heigth
        dv_dt = a               # derivative of the velocity
        return dh_dt, dv_dt

    def RK4_step(self, h:float, v:float, E:float, M:float, Cd:float, dt:float):
        """Solves the SDE wiht a Runge Kutta of fourth order.

        Args:
            h (float): height in the current time 
            v (float): velocity in the current time
            E (float): thrust in the current time
            M (float): mass of the body in the current time
            Cd (float): drag coefficiente in the current time
            dt (float): time step

        Returns:
            float, float: height and velocity in the time step
        """
        k1 = self.diff_eq_system(h, v, E, M, Cd)
        k2 = self.diff_eq_system(h + dt / 2 * k1[0], v + dt / 2 * k1[1], E, M, Cd)
        k3 = self.diff_eq_system(h + dt / 2 * k2[0], v + dt / 2 * k2[1], E, M, Cd)
        k4 = self.diff_eq_system(h + dt * k3[0], v + dt * k3[1], E, M, Cd)

        h_new = h + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v_new = v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return h_new, v_new

    def Euler_step(self, h:float, v:float, E:float, M:float, Cd:float, dt:float):
        """Solves the SDE wiht a Runge Kutta of fourth order.):


        Args:
            h (float): height in the current time 
            v (float): velocity in the current time
            E (float): thrust in the current time
            M (float): mass of the body in the current time
            Cd (float): drag coefficiente in the current time
            dt (float): time step

        Returns:
            float, float: height and velocity in the time step
        """
        dh_dt, dv_dt = self.diff_eq_system(h, v, E, M, Cd)
        h_new = h + dh_dt * dt  
        v_new = v + dv_dt * dt
        return h_new, v_new
    
    def solve_step(self, dt:float=None, E:float=None, M:float=None, Cd:float=None, solver:str=None):
        """solves the SDE in a time step and perfomr the calculation of the atmospheric pressure.
        Properties values of the trayectory are updeted in the atributes of the class.

        Args:
            dt (float, optional): time step. Deafaults to initial class configuration time step
            E (float, optional): thrust in the current time. Defaults to 0.
            M (float, optional): body mass in the current time. Defaults to initial mass in the configuration of the class.
            Cd (float, optional): dragg coefficient in the current time. Defaults to 0.
            solver (str, optional): solver. Defaults to initial solver in class configuration.

        Raises:
            ValueError: invalid solver
        """
        dt = self.dt if dt == None else dt
        E = 0 if E == None else E
        M = self.M0 if M == None else M 
        Cd = 0 if Cd == None else Cd
        solver = self.solver if solver == None else solver

        h = self.h_t[-1]
        v = self.v_t[-1]

        if solver in self.solvers:
            h_new, v_new = self.solvers[solver](h, v, E, M, Cd, dt)
        else:
            raise ValueError(f"Solver '{solver}' is invalid")
        
        (h_new, v_new) = (0, 0) if h_new < 0 else (h_new, v_new) # inelastic ground impact 

        # Results from the SDE 
        self.h_t.append(h_new)
        self.v_t.append(v_new)

        self.t.append(self.t[-1] + dt)

        # Calculating and updating other properties
        self.Pa_t.append(self.atm_properties(h_new)[0])
        

    def flag_stop_condition(self, condition:str, tol=0.1):
        conditions_map = {
        'tf': lambda: self.t[-1] >= self.tf,
        'ground': lambda: self.h_t[-1] < tol and self.v_t[-1] < 0,
        'max_height': lambda: self.v_t[-1] < tol and self.h_t[-1] > tol
        }

        if condition in conditions_map:
            return conditions_map[condition]()
  
    def simulate_trayectory(self, dt:float=None, F:Callable[[float], float]=None, Cd:float=None, tf:float=None, solver:str=None, stop_condition:str='ground'):
        """Performs a simulation of the trayectory by solving sequentyally time steps. Allows to perfom the simulation with different solvers and drag coefficients, 
        as well as setting a thrust funtion along time.

        The stop condition can be set either to a maximum simulation time, or when max height or ground is reached.

        Args:
            dt (float, optional): time step. Deafaults to initial class configuration time step
            F (Callable[[float], float], optional): Thrust function over time. Defaults to 0.
            Cd (float, optional): dragg coefficient. Defaults to 0.
            tf (float, optional): final simulation time. Defaults to None.
            solver (str, optional): solver ('RK4' or 'Euler'). Defaults to the one set in the initial configuration of the class.
            stop_condition (str, optional): stoping condition for the simulation ('tf', 'max_height', 'ground'). Defaults to 'ground'.
        """
        # Restarting the saved values
        self.init_conditions()
        self.init_variables()
        dt = self.dt if dt == None else dt
        self.tf, stop_condition = (tf, 'tf') if tf is not None else (None, stop_condition)
        
        while True:
            E = F(self.t[-1] + dt) if F else None
            self.solve_step(dt, E=E, M=self.M0, Cd=Cd, solver=solver)
            if self.flag_stop_condition(stop_condition): break
                
        self.graphs()
    
    def graphs(self, h=True, v=True, Pa=True, representation_points=50):
        """Generate plot graphs for the properties set to True and scatter a sample of points of the simulation obtained.
        Plots are always graphed with all the data.

        Args:
            h (bool, optional): Enables altitude graph over time. Defaults to True.
            v (bool, optional): Enables velocity graph over time. Defaults to True.
            Pa (bool, optional): Enables atmospheric pressure graph over time. Defaults to True.
            representation_points (int, optional): Number of data to scatter. Defaults to 50.
        """
        def uniform_time_indices(t, representation_points):
            t_min, t_max = t[0], t[-1]
            target_times = np.linspace(t_min, t_max, representation_points)
            indices = [np.abs(t - target_time).argmin() for target_time in target_times]
            return np.array(indices)

        scatter_indices = uniform_time_indices(np.array(self.t), representation_points)
        t_reduced = np.array(self.t)[scatter_indices]

        # Definir configuraciones para cada gráfico en un diccionario
        plots_config = {
            'h': {
                'enabled': h,
                'data': np.array(self.h_t),
                'reduced_data': np.array(self.h_t)[scatter_indices],
                'color': 'blue',
                'label': 'Altura',
                'ylabel': 'h (m)',
                'title': 'Altura en función del tiempo h(t)'
            },
            'v': {
                'enabled': v,
                'data': np.array(self.v_t),
                'reduced_data': np.array(self.v_t)[scatter_indices],
                'color': 'green',
                'label': 'Velocidad',
                'ylabel': 'v (m/s)',
                'title': 'Velocidad en función del tiempo v(t)'
            },
            'Pa': {
                'enabled': Pa,
                'data': np.array(self.Pa_t) / 101325,
                'reduced_data': np.array(self.Pa_t)[scatter_indices] / 101325,
                'color': 'red',
                'label': 'Presión atmosférica',
                'ylabel': 'Pa (atm)',
                'title': 'Presión atmosférica en función del tiempo Pa(t)'
            }
        }

        enabled_plots = [config for config in plots_config.values() if config['enabled']]

        if not enabled_plots:
            print("No graphs enabled.")
            return

        fig, axes = plt.subplots(1, len(enabled_plots), figsize=(6 * len(enabled_plots), 6))

        if len(enabled_plots) == 1:
            axes = [axes]

        for ax, config in zip(axes, enabled_plots):
            ax.plot(self.t, config['data'], label=config['label'])
            ax.scatter(t_reduced, config['reduced_data'], color=config['color'], label="Puntos (reducción)")
            ax.set_xlabel('t (s)')
            ax.set_ylabel(config['ylabel'])
            ax.set_title(config['title'])
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()
if __name__ == '__main__':

    # Crear instancias de las dataclass
    trayectory_geometry_parameters = TrayectoryGeometryParameters(M0=0.5, S=0.003)
    trayectory_physical_constants = TrayectoryPhysicalConstants(g0=9.80665, Ra=287, Re=6.37e6)
    trayectory_initial_conditions = TrayectoryInitialConditions(t0=0, h0=10000, v0=35)
    trayectory_simulation_configuration = TrayectorySimulationConfiguration(dt=2e-05, solver='RK4')

    trayectory1 = trayectory(trayectory_physical_constants, 
                                       trayectory_initial_conditions, 
                                       trayectory_geometry_parameters,
                                       trayectory_simulation_configuration)
    
    trayectory1.simulate_trayectory(dt=100e-5, Cd=2, solver='Euler') 
    