import os
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
import scipy.optimize
import math
from dataclasses import dataclass
from scipy.interpolate import interp1d
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize


@dataclass
class EngineGeometryParameters:
    """Geometry Parameters of the engine."""
    R: float                             # Radius of the chamber
    Rg: float                            # Throat radius
    Rs: float                            # Exit radius
    R0: float                            # Initial fuel radius
    L: float                             # Length of the chamber
    
    def __post_init__(self):
        if any(dimension <= 0 for dimension in [self.R, self.Rg, self.Rs, self.R0, self.L]): 
            raise ValueError("All dimensions must be positive numbers.")

        if self.R0 >= self.R:
            raise ValueError("Initial fuel radious must be smaller than the chamber")
        
        if self.Rs < self.Rg:
            raise ValueError("Exit nozzle radious must be larger than throat radious")
        
        self.As = np.pi * self.Rs ** 2   # Exit Area
        self.Ag = np.pi * self.Rg ** 2   # Throat Area
        self.e = self.As / self.Ag       # Area coefficient

@dataclass
class EngineFuelParameters:
    "Fuel chemical properties"
    Tc: float                           # Combustion Temperature
    a: float                            # a coefficient for recesion velocity
    n: float                            # n coefficient for recesion velocity
    M_molar: float                      # Molar mass of the combusition gas
    gamma: float                        # Gamma constant for the combsution for combustion gas
    rho_pr: float                       # Fuel density
    Rend: float                         # Correction coefficient for the fuel no converted to gas

    def __post_init__(self):
        if any(value <= 0 for value in [self.Tc, self.a, self.n, self.M_molar, self.gamma, self.rho_pr, self.Rend]): 
            raise ValueError("All values must be positive numbers.")
        if self.Rend > 1:
            raise ValueError("Fuel mass to gas effieciency must be between 0 and 1")

        self.Rgas = 8.314 / self.M_molar                                    # Combustion gas constant in SI units
        self.rho_p = self.Rend * self.rho_pr                                # Equivalent density of the real fuel converted to gas 
        self.gammadegamma = np.sqrt(self.gamma) * (2 / (self.gamma + 1)) ** ((self.gamma + 1) / (2 * (self.gamma - 1)))
        self.c_estrella = np.sqrt(self.Rgas * self.Tc) / self.gammadegamma  # C* constant

@dataclass
class EngineInitialConditions:
    """Initial conditions"""
    t0: float                           # Starting time
    Pa0: float                          # Initial atmospheric preassure

    def __post_init__(self):
        if self.Pa0 < 0:
            raise ValueError("Atmospheric pressure must to be a positive or cero value")


@dataclass
class EngineSimulationConfiguration:
    """Additional configurations"""
    dt: float                           # time step 
    solver: str                         # RK4 or Euler
    def __post_init__(self):
        if self.dt <= 0:
            raise ValueError("Time step must be a poistive non zero value")

# Clase RocketEngine que desempaca las dataclass
class RocketEngine:
    """
    Class to determine the thrust over time provided by a rocket engine 
    depending on the geometry caracteristcs of the fuel and nozzle and linked with the exterior through the atmosferic preassure
    """

    def __init__(
            self,
            engine_geometry: EngineGeometryParameters,
            fuel_parameters: EngineFuelParameters,
            initial_condition: EngineInitialConditions,
            simulation_configurations: EngineSimulationConfiguration,
    ):
        # Allowed solvers
        self.solvers = {
            'Euler': self.Euler_step,
            'RK4': self.RK4_step
        }

        # Storing all class attibutes as attibutes in the rocket class
        self.__dict__.update(engine_geometry.__dict__)
        self.__dict__.update(fuel_parameters.__dict__)
        self.__dict__.update(initial_condition.__dict__)
        self.__dict__.update(simulation_configurations.__dict__)

        # Initialiting conditions and constants 
        self.engine_constants()
        self.init_conditions()
        self.init_variables()

        # Initializing geometry values of the rocket for plotting
        self.rocket_contour()

    def engine_constants(self):
        """
        Calculates pressure ratios and mach numbers for chocked flow in the subsonic and supersonic case. 
        Calulates the pressure ratio with a nomal shock wave at the exit. #! Revisar esto de las ondas de choque

        Values are stored as attributes for global access form any point of the class.
        """
        f = lambda M: self.Ag / self.As - M * ((2 + (self.gamma - 1) * M ** 2) / (self.gamma + 1)) ** (- (self.gamma + 1) / (2 * (self.gamma - 1)))
        self.msub = scipy.optimize.fsolve(f, 0)[0] #! CAMBIAR ESTOS EN LOS CODIGOS DE GITHUB
        self.msup = scipy.optimize.fsolve(f, 2)[0]

        self.sub = 1 / (1 + (self.gamma - 1) / 2 * self.msub ** 2) ** ((self.gamma) / (self.gamma - 1))
        self.sup = 1 / (1 + (self.gamma - 1) / 2 * self.msup ** 2) ** ((self.gamma) / (self.gamma - 1))

        m1n = self.msup
        m2n = np.sqrt((2 + (self.gamma - 1) * m1n ** 2) / (2 * self.gamma * m1n ** 2 - (self.gamma - 1)))
        p02_entre_pa = (1 + (self.gamma - 1) / 2 * m2n ** 2) ** (self.gamma / (self.gamma - 1))
        p10_entre_p20 = ((2 * self.gamma * m1n ** 2 - (self.gamma - 1)) / (self.gamma + 1) * ((2 + (self.gamma - 1) * m1n ** 2) / ((self.gamma + 1) * m1n ** 2)) ** self.gamma) ** (1 / (self.gamma - 1))
        self.salto = p02_entre_pa * p10_entre_p20
    
    def init_conditions(self):
        """
        Inits the conditions for the SDE and the lists to store the values
        """
        self.Pc_t = [self.Pa0]      # Chamber pressure along time 
        self.r_t = [self.R0]        # radious along time 
    
    def init_variables(self):
        """Inits lists to store properties of the engine"""
        self.E_t = [0]                                      # Thrust along time        
        self.Mp_t = [self.calculate_propulsant(self.R0)]    # Fuel mass alogn time 
        self.t = [self.t0]                                  # Simulation time

        self.Pa_t = [self.Pa0]                              # Atmospheric pressure over time

    def diff_eq_system(self, Pc:float, r:float, Pa:float):
        """ Differential equation system (SDE) that describes the evolution of the chamber pressure and fuel radious over time 

        Args:
            Pc (float): Chamber preassure in the current instant
            r (float): Interior fuel radious in the current instant
            Pa (float): External chamber preassure in the current instant
        
        Retuns:
            float, float: derivatives of the chamber pressure and fuel radious in the current time

        """
        if r > self.R:                     # with no fuel 
            Rp, Ge, dV = 0, 0, 0 
            Pc = Pa if Pc < Pa else Pc     # in case the download of the tank is so fast to empty all the gas (limited to atmospheric pressure) 
                
        else:
            Rp = self.a * Pc ** self.n      # Recesion velocity
            Ab = 2 * np.pi * self.L * r     # Fuel combustion surface
            Ge = self.rho_p * Rp * Ab       # Combsution mass flow generated 
            dV = Ab * Rp                    # Fuel volume variation (= chamber volume variation) 

        Vc = np.pi * r ** 2 * self.L        # Chamber volume

        X = Pa / self.sub                   # Limit chamber pressure required to chocke the flow
        if Pc < X:
            Gs = self.As * Pa * np.sqrt(self.gamma / (self.Rgas * self.Tc)) * (Pc/Pa) ** ((self.gamma - 1) / (2 * self.gamma)) * np.sqrt(np.abs((2 / (self.gamma - 1)) * ((Pc/Pa) ** ((self.gamma - 1) / self.gamma) - 1))) # Non choqued mass flow exiting the nozzle
            Gs = 0 if Pc < Pa else Gs #! como el Gs se permite que exista aunque Pc sea menor que Pa hay que poner esto para que no se siga descargando cuando no hay presion en la camara
        else:
            Gs = Pc * self.Ag / self.c_estrella                     # Chocked mass flow exiting the nozzle

        dPc_dt = (Ge - Gs) * (self.Rgas * self.Tc - Pc * dV)/ Vc    # Derivitive of the chamber preassure
        dr_dt = Rp                                                  # Derivative of the fuel radious
        return dPc_dt, dr_dt
    
    def RK4_step(self, Pc: float, r:float, Pa:float, dt:float):
        """Runge kutta method of fourth order for solving the SDE of the engine.

        Args:
            Pc (float): Chamber pressure at the current time
            r (float): Fuel raduious at the current time
            Pa (float): Atmospheric pressure outside the engine at the current time
            dt (float): Time step for solving chamber pressure and fuel radious

        Returns:
            float, float: Chamber pressure and fuel radious after the time step
        """

        k1 = self.diff_eq_system(Pc, r, Pa)
        k2 = self.diff_eq_system(Pc + dt / 2 * k1[0], r + dt / 2 * k1[1], Pa)
        k3 = self.diff_eq_system(Pc + dt / 2 * k2[0], r + dt / 2 * k2[1], Pa)
        k4 = self.diff_eq_system(Pc + dt * k3[0], r + dt * k3[1], Pa)

        Pc_new = Pc + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        r_new = r + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return Pc_new, r_new

    def Euler_step(self, Pc: float, r:float, Pa:float, dt:float):
        """Euler method for solving the SDE of the engine.

        Args:
            Pc (float): Chamber pressure at the current time
            r (float): Fuel raduious at the current time
            Pa (float): Atmospheric pressure outside the engine at the current time
            dt (float): Time step for solving chamber pressure and fuel radious

        Returns:
            float, float: Chamber pressure and fuel radious after the time step
        """
        dPc_dt, dr_dt = self.diff_eq_system(Pc, r, Pa)

        Pc_new = Pc + dPc_dt * dt  
        r_new = r + dr_dt * dt
        return Pc_new, r_new

    def solve_step(self, dt:float = None, Pa:float=None, solver:str=None):
        """Provides the solution for the chamber pressure and fuel radius as well as calculating the thrust and fuel mass left after a time step.
           Values are updated in the time list for attributes of the hole class.  

        Args:
            dt (float, optional): time step. Deafaults to initial class configuration time step
            Pa (float, optional): Atmospheric pressure. Defaults to the value provided when init the class.
            solver (int, optional): solver RK4 or Euler. Defaults to the one set at the default of the class when configured.

        Raises:
            ValueError: if solver is not on the list  
        """
        dt = self.dt if dt == None else dt
        Pa = self.Pa0 if Pa == None else Pa
        solver = self.solver if solver == None else solver

        Pc = self.Pc_t[-1]
        r = self.r_t[-1]

        if solver in self.solvers:
            Pc_new, r_new = self.solvers[solver](Pc, r, Pa, dt)
        else:
            raise ValueError(f"Solver '{solver}' no es válido.")
        
        # Updates the solutions of the SDE
        self.t.append(self.t[-1] + dt)
        Pc_new = Pa if Pc_new < Pa else Pc_new  #! revisar esto!!!!
        self.Pc_t.append(Pc_new)
        self.r_t.append(r_new)

        # Calculates and updates engine properties
        self.E_t.append(self.calculate_thrust(Pc_new, Pa))
        self.Mp_t.append(self.calculate_propulsant(r_new))

        self.Pa_t.append(Pa)

    def calculate_thrust(self, Pc:float, Pa:float):
        """Computes the thrust of the engine given the chamber pressure and the exterior one.

        Args:
            Pc (float): Chamber pressure
            Pa (float): Exterior pressure

        Returns:
            float: Thrust genereated
        """

        X = Pa / self.sub
        Z = Pa * self.salto

        if Pc < X:
            Gs = self.As * Pa * np.sqrt(self.gamma / (self.Rgas * self.Tc)) * (Pc/Pa) ** ((self.gamma - 1) / (2 * self.gamma)) * np.sqrt(np.abs((2 / (self.gamma - 1)) * ((Pc/Pa) ** ((self.gamma - 1) / self.gamma) - 1)))
            Gs = 0 if Pc < Pa else Gs #! lo mismo que con el solver porque se permiten Gs cunado Pc < Pa
            Ms = np.sqrt(2 / (self.gamma - 1) * np.abs(((Pc / Pa) ** ((self.gamma -1) / self.gamma) - 1)))
            Ps = Pa
        else:
            Gs = Pc * self.Ag / self.c_estrella

            if Pc < Z:
                Ms = self.msub
                Ps = Pa
            else:
                Ms = self.msup
                Ps = Pc * self.sup

        Ts = self.Tc / (1 + (self.gamma - 1) / 2 * Ms ** 2)
        Vs = Ms * np.sqrt(self.gamma * self.Rgas * Ts)
        E = Gs * Vs + (Ps - Pa) * self.As
        return E
    
    def calculate_propulsant(self, r:float):
        """Calculates the fuel mass left in the chamber.

        Args:
            r (float): interior fuel radious

        Returns:
            float: fuel mass
        """

        Mp = np.pi * (self.R ** 2 - r ** 2) * self.L * self.rho_pr
        return Mp
    
    def flag_stop_condition(self, condition:str, tol=0.1):
        """Determines if a stop condition has been reached within a tolarnace.

        Args:
            condition (str): stoping conditions (final time reached ('tf'), no more fuel and no thrust ('engine_stop'))
            tol (float, optional): tolarance. Defaults to 0.1.

        Returns:
            bool: True if condition is satisfied, False if not
        """
        conditions_map = {
        'tf': lambda: self.t[-1] >= self.tf,                                                # Time has reached tf
        'engine_stop': lambda: self.r_t[-1] > self.R * (1 - tol) and self.E_t[-1] < tol,    # Fuel empty and no thrust
        }

        if condition in conditions_map:
            return conditions_map[condition]()
        else: 
            raise ValueError("Non valid condition entered.")

    def simulate_engine(self, dt:float=None, tf:float=None, Pa:float=None, solver:str=None):
        """Allows to perform simulations by performing several steps of solving the SDE with constant atmospheric pressure.    

        Args:
            dt (float, optional): time step. Deafaults to initial class configuration time step  
            tf (float, optional): final time for the simulation. Defaults to None simulation ending when engine stops.
            Pa (float, optional): external pressure. Defaults to the class atmospheric pressure when configugured.
            solver (_type_, optional): _description_. Defaults to None.
        """
        Pa = self.Pa0 if Pa == None else Pa
        self.tf, stop_condition = (tf, 'tf') if tf is not None else (None, 'engine_stop')

        # Resets saved values
        self.init_conditions()  
        self.init_variables()

        while True:
            self.solve_step(dt=dt, Pa=Pa, solver=solver)
            if self.flag_stop_condition(stop_condition): break

        self.graphs()

    def graphs(self, Pc=True, r=True, E=True, representation_points=50):
        """Generate plot graphs for the properties set to True and scatter a sample of points of the simulation obtained.
        Plots are always graphed with all the data.

        Args:
            Pc (bool, optional): Enables chamber pressure graph over time. Defaults to True.
            r (bool, optional): Enables fuel radius graph along time. Defaults to True.
            E (bool, optional): Enables thrust graph along time. Defaults to True.
            representation_points (int, optional): Number of data to scatter. Defaults to 50.
        """
        scatter_indices = np.linspace(0, len(self.t) - 1, representation_points, dtype=int)
        t_reduced = np.array(self.t)[scatter_indices]

        plots_config = {
            'Pc': {
                'enabled': Pc,
                'data': np.array(self.Pc_t) / 101325,
                'reduced_data': np.array(self.Pc_t)[scatter_indices] / 101325,
                'color': 'red',
                'label': 'Presión de la cámara (atm)',
                'ylabel': 'Pc (atm)',
                'title': 'Evolución de la presión en la cámara'
            },
            'r': {
                'enabled': r,
                'data': np.array(self.r_t) * 100,
                'reduced_data': np.array(self.r_t)[scatter_indices] * 100,
                'color': 'green',
                'label': 'Radio',
                'ylabel': 'r (cm)',
                'title': 'Evolución del radio del combustible a lo largo del tiempo'
            },
            'E': {
                'enabled': E,
                'data': np.array(self.E_t),
                'reduced_data': np.array(self.E_t)[scatter_indices],
                'color': 'blue',
                'label': 'Empuje',
                'ylabel': 'E (N)',
                'title': 'Empuje en función del tiempo'
            }
        }

        enabled_plots = [config for config in plots_config.values() if config['enabled']]
        if not enabled_plots:
            print("No graphs enabled")
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
    
    ######### calculos de las propiedades en las distintas aereas de la tobera ######################
    @lru_cache(maxsize=None)
    def calculate_m1n(self, Pc:float, Pa:float):
        """Calculates the mach number before a shockwave inside the nozzle por a given chamber and atmospheric pressure

        Args:
            Pc (_type_): chamber pressure
            Pa (_type_): atmospheric pressure

        Returns:
            float: mach number before the shockwave
        """
        return scipy.optimize.brentq(self.eq_NSW, 0.9, self.msup + 0.1, args=(Pc, Pa))

    def calculate_A_NSW(self, Pc:float, Pa:float):
        """Calculates the area of the nozzle where normal shockwave (NSW) occurs for a given chamber and atmospheric pressures. 

        Args:
            Pc (float): chamber pressure
            Pa (float): atmospheric pressure

        Returns:
            float: area of the nozzle where shockwave
        """
        m1n = self.calculate_m1n(Pc, Pa) # mach number before shockwave inside nozzle
        A_NSW = self.Ag / (m1n * ((2 + (self.gamma - 1) * m1n ** 2) / (self.gamma + 1)) ** (-(self.gamma + 1) / (2 * (self.gamma - 1))))
        return A_NSW
    
    def eq_NSW(self, m1n: float, Pc:float, Pa:float):
        """ Non linear implict equation that relates mach number before the showave (m1n) with a chamber and atmospheric pressures. 
        Type f(m1n) = 0.

        Args:
            m1n (float): mach number before the shockwave
            Pc (float): chamber pressure
            Pa (float): atmospheric pressure

        Returns:
            float: left side of the equation f(m1n).
        """
        m2n = np.sqrt((2 + (self.gamma - 1) * m1n ** 2) / (2 * self.gamma * m1n ** 2 - (self.gamma - 1)))
        p10_entre_p20 = ((2 * self.gamma * m1n ** 2 - (self.gamma - 1)) / (self.gamma + 1) * ((2 + (self.gamma - 1) * m1n ** 2) / ((self.gamma + 1) * m1n ** 2)) ** self.gamma) ** (1 / (self.gamma - 1))
        p02_entre_pa = Pc / Pa / p10_entre_p20
        argument = 2 / (self.gamma - 1) * (p02_entre_pa ** ((self.gamma - 1) / self.gamma) - 1)
        ms = np.sqrt(argument) if argument >= 0 else 1e06
        A_NSW = self.Ag / (m1n * ((2 + (self.gamma - 1) * m1n ** 2) / (self.gamma + 1)) ** (-(self.gamma + 1) / (2 * (self.gamma - 1))))
        return A_NSW / self.As - ms / m2n * ((2 + (self.gamma - 1) * ms ** 2) / (2 + (self.gamma - 1) * m2n ** 2)) ** (-(self.gamma + 1) / (2 * (self.gamma - 1)))

    def eq_Ms(self, M:float, A:float, A_star:float):
        """Non linear implicit equation that relates the mach number at a given section of the nozzle of section area A with the area of the critical section A*.
            This equation allows to find the mach number corresponding to a given area. 
            Note that this equation has two solutions for a given A*/A value one for subsonic and other for supersonic case.
        Args:
            M (_type_): mach number at section area A
            A (_type_): section area of the nozzle with mach number M
            A_star (_type_): critical area (smaller than the all of the sections if non chocked flow or equla to the minimum area, the throat, in case of chocked flow).

        Returns:
            _type_: _description_
        """
        return A_star / A - M * ((2 + (self.gamma - 1) * M ** 2) / (self.gamma + 1)) ** (- (self.gamma + 1) / (2 * (self.gamma - 1)))

    def section_properties(self, A:float, y_A:float, Pc:float, Pa:float):
        """Calculates the fluid properties: mach number, pressure and temperature at a section A of the nozzle for a given chamber and atmospheric pressures.
            It is assumed that nozzle is oriented vertically with the divergent section facing down. 
            The vertical coordinate of the throat provides the orientation as section areas with higher y coordinates are considered in the convergent region.
    
        Args:
            A (float): area of the nozzle section to obtain the properties
            Pc (float): chmaber pressure
            Pa (float): atmospheri pressure at the exit of the nozzle.
            y (float): heigth of the throat

        Returns:
            _type_: _description_
        """
        X = Pa / self.sub           # Chamber pressure limit for chocked the flow
        Z = Pa * self.salto         # Chamber pressure limit for normal shock wave (NSW) at the exit
        Ms = np.sqrt(2 / (self.gamma - 1) * np.abs(((Pc / Pa) ** ((self.gamma -1) / self.gamma) - 1)))
        if y_A > self.y_g:  ## Convergent nozzle section
                if Pc < X:  #  Non choked flow               
                    Ms = np.sqrt(2 / (self.gamma - 1) * ((Pc / Pa) ** ((self.gamma -1) / self.gamma) - 1))                                       # ec 10.7.11
                    A_star = self.As * Ms * ((2 + (self.gamma - 1) * Ms **2) / (self.gamma + 1)) ** (- (self.gamma + 1) / (2 * (self.gamma -1))) # ec 10.7.12
                    M = scipy.optimize.newton(self.eq_Ms, 0, args=(A, A_star))  # non chocked ans subsonic solution
                else:       # Chocked flow
                    M = scipy.optimize.newton(self.eq_Ms, 0, args=(A, self.Ag)) # (A* = Ag) chocked and subsoinc solution 
        
        else:               ## Divergent nozzle section
            if Pc < X:      #  Non chocked flow
                Ms = np.sqrt(2 / (self.gamma - 1) * ((Pc / Pa) ** ((self.gamma -1) / self.gamma) - 1))                                           # ec 10.7.11
                A_star = self.As * Ms * ((2 + (self.gamma - 1) * Ms **2) / (self.gamma + 1)) ** (- (self.gamma + 1) / (2 * (self.gamma -1)))     # ec 10.7.12
                M = scipy.optimize.newton(self.eq_Ms, 0, args=(A, A_star))      # A* = Ag

            else:           # Chocked flow
                if Pc < Z:  # NSW inside nozzle
                    A_lim = self.calculate_A_NSW(Pc, Pa)
                    m1n = self.calculate_m1n(Pc, Pa)
                    if A < A_lim: # supersonic before NSW
                        M = scipy.optimize.newton(self.eq_Ms, 2, args = (A, self.Ag)) # chocked (A* = Ag) and subsonic solution

                    else:   # subsonic after NSW
                        p10_entre_p20 = ((2 * self.gamma * m1n ** 2 - (self.gamma - 1)) / (self.gamma + 1) * ((2 + (self.gamma - 1) * m1n ** 2) / ((self.gamma + 1) * m1n ** 2)) ** self.gamma) ** (1 / (self.gamma - 1))
                        p02_entre_pa = Pc / Pa / p10_entre_p20
                        Ms = np.sqrt(2 / (self.gamma - 1) * (p02_entre_pa ** ((self.gamma - 1) / self.gamma) - 1 ))
                        A_star = self.As * Ms * ((2 + (self.gamma - 1) * Ms **2) / (self.gamma + 1)) ** (- (self.gamma + 1) / (2 * (self.gamma -1))) # ec 10.7.12
                        M = scipy.optimize.newton(self.eq_Ms, 0, args=(A, A_star)) # subsonic solution

                else:      # NSW outside nozzle, all divergent nozzle region supersonic       
                    M = scipy.optimize.newton(self.eq_Ms, 2, args = (A, self.Ag))  # supersonic solution

        P = Pc / (1 + (self.gamma - 1) / 2 * M ** 2) ** ((self.gamma) / (self.gamma - 1))       # pressure for isentropric flow
        T = self.Tc / (1 + (self.gamma - 1) / 2 * M ** 2) ** ((self.gamma) / (self.gamma - 1))  # temperatur for isentropc flow
        return M, P, T
    #################################################################################################
    def nozzle_contour_grid(self, x_points: int, y_points: int):
        """
        Creates a grid for the nozzle.

        Args:
            x_points (int): number of points for the grid along x direction.
            y_points (int): nubmer of poinst for the grid along y direction.

        Returns:
            np.ndarray, np.ndarray, np.ndarray: X nozzle grid, Y nozzle grid, mask (boolean matrix that represents which points are inside the limits of the nozzle).
        """
        nozzle_mesh_y = np.linspace(self.y.max(), self.y.min(), x_points)
        nozzle_mesh_x = np.linspace(min(self.x_), max(self.x), y_points)
        X_grid, Y_grid = np.meshgrid(nozzle_mesh_x, nozzle_mesh_y)

        mask = np.zeros_like(X_grid, dtype=bool)
        x_left_grid = self.interp_left(Y_grid)
        x_right_grid = self.interp_right(Y_grid)
        mask = (X_grid >= x_left_grid) & (X_grid <= x_right_grid)

        return X_grid, Y_grid, mask

    def graph_rocket(self, Pc:float=None, Pa:float=None, r=None, property=None, resolx:int=100, resoly:int=100):
        """
        graphs the rocket engine geometry, additional arguments can add contour for gass properties inside the engine, 
        fuel radious can be specified in a value if given or if not will equal initial radious.

        Args:
            Pc (float, optional): chammber pressure. Defaults to None if only geometry wants to be plotted.
            Pa (float, optional): atomspheric pressure. Defaults to None if only geometry wants to be plotted.
            r (float, optional): radious of the inside of the fuel. Default value equal to the starting value.
            property (str, optional): dictionary key for the property to plot contour ('M': mach number, 'T': temperature, 'P': pressure).
                                    Default None, no contour graph only geometry.
            resolx (int, optional): x resolution of the nozzle contour mesh. Higher values allow better fitting to the edges of the nozzle.
                                    Default 100.
            resoly (int, optional): y resolutino of th enozzle conour mesh. Higher valus allow smoother colour gradients
                                    Default 100.
        """

        properties = {'M': 'Mach number', 'P': 'Pressure', 'T': 'Temperature'}
        cmap, norm = None, None

        # checking input values can be possible 
        if property is not None and property not in properties: raise ValueError(f"Property key not found. Valid keys are: {', '.join(properties.keys())}")
        r = self.R0 if r == None else r
        if r > self.R or r< self.R0: raise ValueError("r is outside limits")

        ## when contour plot is asked
        if property is not None:
            if Pc == None or Pa == None: raise("Pc and Pa must be provided for contour plot")
            chamber_values = {'M': 0, 'P': Pc / 101325, 'T': self.Tc}
            X_grid, Y_grid, mask = self.nozzle_contour_grid(resolx, resoly)
            cmap, norm = self.cmaps[property], self.norms[property]
        else:
            X_grid, Y_grid, mask = None, None, None
        values = None if property is None else self.y_nozzle_values(Y_grid, mask, Pc, Pa, property=property)

        # Graph 
        fig, ax = plt.subplots()
        ax = self.plot_rocket_contour(ax, r, X_grid, Y_grid, chamber_values[property], values, cmap=cmap, norm=norm)
        ax.set_title(self.properties_titles[property])
        plt.show()
    
    def bell_nozzle(self, k, aratio, Rt, l_percent):
        entrant_angle = -135
        ea_radian = math.radians(entrant_angle)

        if l_percent == 60:
            Lnp = 0.6
        elif l_percent == 80:
            Lnp = 0.8
        elif l_percent == 90:
            Lnp = 0.9
        else:
            Lnp = 0.8

        angles = self.find_wall_angles(aratio, Rt, l_percent)
        nozzle_length = angles[0]
        theta_n = angles[1]
        theta_e = angles[2]

        data_intervel = 100
        ea_start = ea_radian
        ea_end = -math.pi / 2
        angle_list = np.linspace(ea_start, ea_end, data_intervel)
        xe = []
        ye = []
        for i in angle_list:
            xe.append(1.5 * Rt * math.cos(i))
            ye.append(1.5 * Rt * math.sin(i) + 2.5 * Rt)

        ea_start = -math.pi / 2
        ea_end = theta_n - math.pi / 2
        angle_list = np.linspace(ea_start, ea_end, data_intervel)
        xe2 = []
        ye2 = []
        for i in angle_list:
            xe2.append(0.382 * Rt * math.cos(i))
            ye2.append(0.382 * Rt * math.sin(i) + 1.382 * Rt)

        Nx = 0.382 * Rt * math.cos(theta_n - math.pi / 2)
        Ny = 0.382 * Rt * math.sin(theta_n - math.pi / 2) + 1.382 * Rt
        Ex = Lnp * ((math.sqrt(aratio) - 1) * Rt) / math.tan(math.radians(15))
        Ey = math.sqrt(aratio) * Rt
        m1 = math.tan(theta_n)
        m2 = math.tan(theta_e)
        C1 = Ny - m1 * Nx
        C2 = Ey - m2 * Ex
        Qx = (C2 - C1) / (m1 - m2)
        Qy = (m1 * C2 - m2 * C1) / (m1 - m2)

        int_list = np.linspace(0, 1, data_intervel)
        xbell = []
        ybell = []
        for t in int_list:
            xbell.append(((1 - t) ** 2) * Nx + 2 * (1 - t) * t * Qx + (t ** 2) * Ex)
            ybell.append(((1 - t) ** 2) * Ny + 2 * (1 - t) * t * Qy + (t ** 2) * Ey)

        nye = [-y for y in ye]
        nye2 = [-y for y in ye2]
        nybell = [-y for y in ybell]

        return angles, (xe, ye, nye, xe2, ye2, nye2, xbell, ybell, nybell)

    def find_wall_angles(self, ar, Rt, l_percent=80):
        aratio = [4, 5, 10, 20, 30, 40, 50, 100]
        theta_n_60 = [20.5, 20.5, 16.0, 14.5, 14.0, 13.5, 13.0, 11.2]
        theta_n_80 = [21.5, 23.0, 26.3, 28.8, 30.0, 31.0, 31.5, 33.5]
        theta_n_90 = [20.0, 21.0, 24.0, 27.0, 28.5, 29.5, 30.2, 32.0]
        theta_e_60 = [26.5, 28.0, 32.0, 35.0, 36.2, 37.1, 35.0, 40.0]
        theta_e_80 = [14.0, 13.0, 11.0, 9.0, 8.5, 8.0, 7.5, 7.0]
        theta_e_90 = [11.5, 10.5, 8.0, 7.0, 6.5, 6.0, 6.0, 6.0]

        f1 = ((math.sqrt(ar) - 1) * Rt) / math.tan(math.radians(15))

        if l_percent == 60:
            theta_n = theta_n_60
            theta_e = theta_e_60
            Ln = 0.6 * f1
        elif l_percent == 80:
            theta_n = theta_n_80
            theta_e = theta_e_80
            Ln = 0.8 * f1
        elif l_percent == 90:
            theta_n = theta_n_90
            theta_e = theta_e_90
            Ln = 0.9 * f1
        else:
            theta_n = theta_n_80
            theta_e = theta_e_80
            Ln = 0.8 * f1

        x_index, x_val = self.find_nearest(aratio, ar)
        if round(aratio[x_index], 1) == round(ar, 1):
            return Ln, math.radians(theta_n[x_index]), math.radians(theta_e[x_index])

        if x_index > 2:
            ar_slice = aratio[x_index - 2 : x_index + 2]
            tn_slice = theta_n[x_index - 2 : x_index + 2]
            te_slice = theta_e[x_index - 2 : x_index + 2]
            tn_val = self.interpolate(ar_slice, tn_slice, ar)
            te_val = self.interpolate(ar_slice, te_slice, ar)
        elif len(aratio) - x_index <= 1:
            ar_slice = aratio[x_index - 2 : len(x_index)]
            tn_slice = theta_n[x_index - 2 : len(x_index)]
            te_slice = theta_e[x_index - 2 : len(x_index)]
            tn_val = self.interpolate(ar_slice, tn_slice, ar)
            te_val = self.interpolate(ar_slice, te_slice, ar)
        else:
            ar_slice = aratio[0 : x_index + 2]
            tn_slice = theta_n[0 : x_index + 2]
            te_slice = theta_e[0 : x_index + 2]
            tn_val = self.interpolate(ar_slice, tn_slice, ar)
            te_val = self.interpolate(ar_slice, te_slice, ar)

        return Ln, math.radians(tn_val), math.radians(te_val)

    def interpolate(self, x_list, y_list, x):
        if any(y - x <= 0 for x, y in zip(x_list, x_list[1:])):
            raise ValueError("x_list must be in strictly ascending order!")
        intervals = zip(x_list, x_list[1:], y_list, y_list[1:])
        slopes = [(y2 - y1) / (x2 - x1) for x1, x2, y1, y2 in intervals]

        if x <= x_list[0]:
            return y_list[0]
        elif x >= x_list[-1]:
            return y_list[-1]
        else:
            i = self.bisect_left(x_list, x) - 1
            return y_list[i] + slopes[i] * (x - x_list[i])

    def bisect_left(self, a, x, lo=0, hi=None, *, key=None):
        """Return the index where to insert item x in list a, assuming a is sorted.

        The return value i is such that all e in a[:i] have e < x, and all e in
        a[i:] have e >= x.  So if x already appears in the list, a.insert(i, x) will
        insert just before the leftmost x already there.

        Optional args lo (default 0) and hi (default len(a)) bound the
        slice of a to be searched.
        """

        if lo < 0:
            raise ValueError('lo must be non-negative')
        if hi is None:
            hi = len(a)
        # Note, the comparison uses "<" to match the
        # __lt__() logic in list.sort() and in heapq.
        if key is None:
            while lo < hi:
                mid = (lo + hi) // 2
                if a[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
        else:
            while lo < hi:
                mid = (lo + hi) // 2
                if key(a[mid]) < x:
                    lo = mid + 1
                else:
                    hi = mid
        return lo
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]


    def rocket_contour(self):
        """" #! ADD function description
        """ 
        angles, contour = self.bell_nozzle(1.21, self.e, self.Rg, 0.8)
        ye, xe, _, ye2, xe2, _, ybell, xbell, _ = contour

        ## Sections of the nozzle
        ye = np.array([-y for y in ye])         # convergent region
        ye2 = np.array([-y for y in ye2])       # transition region
        ybell = np.array([-y for y in ybell])   # divergent region bell

        ye, ye2, ybell = ye - np.min(ybell), ye2 - np.min(ybell), ybell - np.min(ybell) # transalation of the exit of the nozzle to the origin

        # Saving values as attributes
        self.x = np.concatenate((xe, xe2, xbell))   
        self.y = np.concatenate((ye, ye2, ybell))   
        self.x_ = [-xi for xi in self.x]            # simetry side
        self.y_g = self.y[np.abs(self.x).argmin()]  # finding y coordinate for throat

        # Interpolation functions for the contour
        self.interp_left = interp1d(self.y, self.x_, kind='linear', fill_value='extrapolate')
        self.interp_right = interp1d(self.y, self.x, kind='linear', fill_value='extrapolate')

        # Limit points of the chamber geometry
        point1 = (np.max(xe), np.max(ye))             # ending of the convergent region of the nozzle
        point2 = (self.R, point1[1])                  # right bottom corner of the chamber
        point3 = (point2[0], point2[1] + self.L)      # right top corner of he chamber
        point4 = (0, point3[1])                       # middle top point of the chamber          

        self.y_fuel= [point1[1], point1[1], point3[1], point3[1]] # y coordinates of the fuel

        self.segment1 = {'x': [point1[0], point2[0]], 'y': [point1[1], point2[1]]}      # bottom right horizontal  line of chamber
        self.segment2 = {'x': [point2[0], point3[0]], 'y': [point2[1], point3[1]]}      # right vertical line of the chamber
        self.segment3 = {'x': [point3[0], point4[0]], 'y': [point3[1], point4[1]]}      # top right horizontal  line of the chamber

        self.segment1_ = {'x': [-point1[0], -point2[0]], 'y': [point1[1], point2[1]]}   # bottom left horizontal  line of chamber
        self.segment2_ = {'x': [-point2[0], -point3[0]], 'y': [point2[1], point3[1]]}   # left vertical line of the chamber
        self.segment3_ = {'x': [-point3[0], -point4[0]], 'y': [point3[1], point4[1]]}   # top left horizontal  line of the chamber
        
        colors_M = [
            (0, "darkblue"),   
            (0.25, "blue"),   
            (0.5, "cyan"),
            (0.75, "green"),
            (1, "yellow"),  
            (2, "orange"), 
            (3, "red"),  
        ]

        colors_P = [ 
            (0, "darkblue"),        # Pressure values in bar 
            (1, "blue"),   
            (2, "cyan"),
            (3, "green"),
            (5, "yellow"),  
            (10, "orange"), 
            (25, "red"),  
        ]

        colors_T = [                
            (0, "darkblue"),        # Temperature values in K
            (100, "midnightblue"),        
            (200, "blue"),
            (300, "deepskyblue"),         
            (400, "cyan"),
            (500, "aquamarine"),          
            (600, "green"),
            (700, "limegreen"),           
            (800, "yellow"),
            (900, "gold"),                
            (950, "orange"),
            (1000, "red"),
        ]

        cmap_M, norm_M = self.create_custom_colormap(colors_M)
        cmap_P, norm_P = self.create_custom_colormap(colors_P)
        cmap_T, norm_T = self.create_custom_colormap(colors_T)

        self.norms = {'M': norm_M, 'P': norm_P, 'T': norm_T}
        self.cmaps = {'M': cmap_M, 'P': cmap_P, 'T': cmap_T}

        self.properties_titles = {'M': 'Mach number', 'P': 'Pressure (atm)', 'T': 'Temperature (K)'}
        
    def plot_rocket_contour(self, ax, r:float, X_grid, Y_grid, chamber_value=None, values=None, cmap=None, norm=None):
        """Add to an existent figure axis the geometry plot of the engine and a contour plot in the nozzle and chamber
        in case values for the plot are provided.

        Args:
            ax (plt.axis): axis of an existent figure
            r (float): interior fuel radious
            X_grid (np.ndarray): nozzle grid for the contour plot
            Y_grid (np.ndarray): nozzle grid for the contour plot
            chamber_value (float, optional): value of the property in the chamber. Default None, not necessay without contour plot.
            values (np.ndarray, optional): values in each point of the grid representing a property wiht np.nan outside nozzle limits.
              Defaults to None, in only geometry wants to be plotted.
            cmap (mpl.colors.LinearSegmentedColormap, optional): color map to plot the contour values.
            norm (mpl.colors.LinearSegmentedColormap, optional): norm for the color map representation. See rocket_contour().
        """

        x_fuel = [r, self.R, self.R, r]
        x_fuel_ = [-xi for xi in x_fuel]

        ax.clear()
        ax.plot(self.x, self.y, label='Tobera', color='black')
        ax.plot(self.x_, self.y, label='Tobera', color='black')

        ax.plot(self.segment1['x'], self.segment1['y'], color='black')
        ax.plot(self.segment2['x'], self.segment2['y'], color='black')
        ax.plot(self.segment3['x'], self.segment3['y'], color='black')
        ax.plot(self.segment1_['x'], self.segment1_['y'], color='black')
        ax.plot(self.segment2_['x'], self.segment2_['y'], color='black')
        ax.plot(self.segment3_['x'], self.segment3_['y'], color='black')

        if values is not None:
            x_chamber = [x_fuel_[0], x_fuel[0], x_fuel[0], x_fuel_[0]]
            y_chamber = np.copy(self.y_fuel)
            y_chamber[0] += self.L / 10
            y_chamber[1] += self.L / 10
            ax.fill(x_chamber, y_chamber, color=cmap(norm(chamber_value)))

            xlim = [x_fuel_[0], x_fuel[0]]
            ylim = [self.y_fuel[0], y_chamber[0]]

            nozzle_top_val = np.array(values[0,:])[~np.isnan(np.array(values[0,:]))][0]
            self.fill_rectangle_with_mesh(ax, xlim, ylim, chamber_value, nozzle_top_val, cmap, norm)

            im = ax.pcolormesh(X_grid, Y_grid, values, cmap=cmap, norm=norm, shading='auto')
            
            try: self.cbar.remove()
            except: pass
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)  # Tamaño fijo para la barra de color
            self.cbar = plt.colorbar(im, cax=cax)


        ax.fill(x_fuel, self.y_fuel, color='grey')
        ax.fill(x_fuel_, self.y_fuel, color='grey')
        ax.set_aspect('equal')
        plt.ylabel('Longitudinal axis (m)')
        return ax
    
    def fill_rectangle_with_mesh(self, ax, xlim, ylim, bottom_val, top_val, cmap, norm, resolutionx=2, resolutiony=200):
        """Adds to an exitent axis a plot to fill with a smooth transition a squared to join the inisde of the chamber wiht the start of the nozzle.

        Args:
            ax (plt.axis): axis to plot the contour
            xlim (list): list with the x limits of the square
            ylim (list): list with the y limits of the square
            bottom_val (float): value of the property in the start of the nozzle
            top_val (float): value of the property in the chamber
            cmap (plt.cmap): color map used for the contour in the nozzle.
            norm (_type_): norm used for the colormap in the nozzle.
            resolutionx (int, optional): number of points on x direction. Defaults to 20.
            resoltuiony (int, optional): number of poinst on y driection. Higher improves the quality of the transition. Defaults 100.
        """
        x = np.linspace(xlim[0], xlim[1], resolutionx)
        y = np.linspace(ylim[0], ylim[1], resolutiony)
        X, Y = np.meshgrid(x, y)
        
        # Interpolar linealmente los valores desde el borde superior (top_val) al inferior (bottom_val)
        gradient = np.linspace(top_val, bottom_val, y.size).reshape(-1, 1)
        
        # Aplicar el colormap y la normalización al gradiente
        colored_gradient = cmap(norm(gradient))
        
        # Visualizar el resultado
        ax.imshow(colored_gradient, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto')
        ax.set_title('Rectángulo con Gradiente Basado en Coordenada Vertical')


    def engine_animation(self, grid_x_points: int=100, grid_y_points: int=100, property: str='M', t_points: int=500, 
                         t0_percentage: float=0, tf_percentage: float=1, speed_multiplier: float=1, save:bool=0):
        """Animation for the engine beahivour during time including the espatial evolution over time of a the mach number, pressure or temperature.
        Args:
            x_points (int, optional): number of x points for nozzle grid. Defaults to 100.
            y_points (int, optional): number of y points for nozzle grid. Defaults to 100.
            property (str): property to obtain in y coordinate of the nozzle. 'M', 'P', 'T' (mach number, temperature or density) Defaults to 'M'.
            t_points (int, optional): number of time points to represent the animation. Defaults to 500.
            t0_percentage (int): percentage of total time to start data calculation (0, 1). Defaults to 0.
            tf_percetage (int): percentage of total time to finsih data calculation (0, 1). Defaults to 1.
            speed_multiplier (float): multplier of speed rate at which the animation is going to play. 
                                    Defualts equals to 1, real time.
            save (bool): if True saves a .gif of the animation.
        """

        X_grid, Y_grid, mask = self.nozzle_contour_grid(grid_x_points, grid_y_points)
        data = self.animation_data(Y_grid, mask, property, t_points, t0_percentage, tf_percentage)

        t = data['t']
        r = data['r']
        pc = data['pc']
        pa = data['pa']
        grid_values_all = data['grid_values']
        chamber_values = {'M': np.zeros_like(t), 'P': pc, 'T': self.Tc * np.ones_like(t)}
        cmap , norm = self.cmaps[property], self.norms[property]
        fig, ax = plt.subplots()

        def update_frame(i):
            self.plot_rocket_contour(ax, r[i], X_grid, Y_grid, chamber_values[property][i], grid_values_all[i], cmap=cmap, norm=norm)
            ax.set_title(self.properties_titles[property])
            ax.set_xlabel(str(round(t[i], 2)) + 's')
            return ax
        
        interval = (t[-1] - t[0]) * 1000 / len(t) * speed_multiplier
        anim = animation.FuncAnimation(fig, update_frame, frames=len(t), interval=interval)
        plt.show()

        if save == True:
            fps = 1000 / interval
            if not os.path.exists('animation_data'): os.makedirs('animation_data') 
            anim.save('animation_data/animacion.gif', writer='imagemagick', fps=fps, dpi=300)

    def create_custom_colormap(self, colors:list):
        """creates a custom colormap from a color list containg relationships between values and colors.
        colors list must be a list of tupples where first element of each tuple contains the value and the seconde elment the color,
        values should be sorted in ascendent order so that first tuple contains the color for the min value or min limit values
        adn the last one the relationship betweeen the highest value limit and its color.

        Args:
            colors (list): list of tuples containing relationships between values and their colors.

        Returns:
            mpl.colors.LinearSegmentedColormap: color map to plot the contour values.
            mpl.colors.LinearSegmentedColormap: norm for the color map representation. See rocket_contour().
        """
        vmin = colors[0][0]
        vmax = colors[-1][0]
        norm = Normalize(vmin, vmax)
        colors = [[norm(colors[i][0]), colors[i][1]] for i in range(len(colors))]
        custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)
        return custom_cmap, norm
    
    def y_section_property_value(self, y_coord:float, Pc:float, Pa:float, property:str):
        """Returns the property value given (M, P, T) in a nozzle y coordinate.

        Args:
            y_coord (float): y coordinate of a nozzle point 
            Pc (float): chamber pressure
            Pa (float): atmospheric pressure
            property (str): property to obtain in y coordinate of the nozzle. 'M', 'P', 'T' (mach number, temperature or density)

        Returns:
            float: value of the property to obtian in coordinate y of the nozzle.
        """
        r = self.interp_right(y_coord)
        A = np.pi * r ** 2
        M, P, T = self.section_properties(A, y_coord, Pc, Pa)
        properties_values = {'M': M, 'P': P / 101325, 'T': T}
        return properties_values[property]

    def y_nozzle_values(self, Y_grid: np.ndarray, mask: np.ndarray, Pc: float, Pa: float, property: str):
        """calculates the given property value along the y direction of the nozzle grid for a chamber and atomspheric pressure.

        Args:
            Y_grid (numpy.ndarray): y mesh of nozzle (squared)
            mask (numpy.ndarray): array with boolean values where only values inside nozzle contour are True.
            Pc (float): chamber pressure
            Pa (float): atmospheric pressure
            property (str): property to obtain in y coordinate of the nozzle. 'M', 'P', 'T' (mach number, temperature or density)

        Returns:
            np.ndarray: mesh with values of the property in each point of the mesh (outside of the nozzle np.nan)
        """
        vectorized_color_function = np.vectorize(self.y_section_property_value, excluded=['Pc', 'Pa', 'property'])
        nozzle_mesh_y = np.unique(Y_grid.flatten()).tolist()[::-1]
        values_along_y= vectorized_color_function(nozzle_mesh_y, Pc=Pc, Pa=Pa, property=property)
        m, n = Y_grid.shape
        nozzle_mesh_values = np.tile(values_along_y.reshape(m, 1), (1, n)) 
        nozzle_mesh_values = np.where(mask, nozzle_mesh_values, np.nan)     # only values inside nozzle area (from mask == True)
        return nozzle_mesh_values
    
    def animation_data(self, Y_grid:np.ndarray, mask:np.ndarray, property:str, t_points:int, t0_percentage: float, tf_percentage: float):
        """Calculates the data necessary for the contour plot.
        Values of the property are obtained for every point of the grid of the nozzle and each timestep.

        Args:
            Y_grid (np.ndarray): nozzle grid Y for the contour
            mask (np.ndarray): matrix that determines which points of the grid are inside of the nozzle limits
            property (str): property to obtain in y coordinate of the nozzle. 'M', 'P', 'T' (mach number, temperature or density)
            t_points (_type_): number of points in which to reduce the time vector.
            t0_percentage (int): percentage of total time to start data calculation.
            tf_percetage (int): percentage of total time to finsih data calculation.
        Returns:
            dict: data for animtion: t': t_reduced, 'r': r_reduced, 'pc': pc_reduced, 'pa':pa_reduced, 'grid_values':grid_values_list
        """

        # Reducing the time vector to t_points steps
        scatter_indices = np.linspace(int(t0_percentage * (len(self.t) - 1)), int(tf_percentage * (len(self.t) - 1)), t_points, dtype=int)
        t_reduced = np.array(self.t)[scatter_indices]
        pc_reduced = np.array(self.Pc_t)[scatter_indices] 
        pa_reduced = np.array(self.Pa_t)[scatter_indices] 
        r_reduced = np.array(self.r_t)[scatter_indices] 

        # Calculating the values of the property for each time instant
        grid_values_list = []
        for index in range(len(t_reduced)):  
            grid_values = self.y_nozzle_values(Y_grid, mask, pc_reduced[index], pa_reduced[index], property)
            grid_values_list.append(grid_values)

        data ={'t': t_reduced, 'r': r_reduced, 'pc': pc_reduced / 101325, 'pa': pa_reduced / 101325, 'grid_values': grid_values_list} 
        return data
        
if __name__ == '__main__':
    # Crear instancias de las dataclass
    engine_geometry = EngineGeometryParameters(R=2.25e-2, Rg=0.5e-2, Rs=0.5e-2, R0=0.5e-2, L=8e-2)
    fuel_parameters = EngineFuelParameters(Tc=1000, a=6e-5, n=0.32, M_molar=41.98e-3, gamma=1.3, rho_pr=1800, Rend=1 - 0.4237)
    initial_conditions = EngineInitialConditions(t0=0, Pa0=101325)
    simulation_configurations = EngineSimulationConfiguration(solver='RK4', dt=2e-05)
    
    # Crear instancia de RocketEngine y pasar las instancias de las dataclass
    rocket_engine = RocketEngine(engine_geometry, fuel_parameters, initial_conditions, simulation_configurations)
    # rocket_engine.simulate_engine()


    engine_geometry2 = EngineGeometryParameters(R=2.25e-2, Rg=0.5e-2, Rs=2e-2, R0=1e-2, L=8e-2)
    rocket_engine2 = RocketEngine(engine_geometry2, fuel_parameters, initial_conditions, simulation_configurations)

    rocket_engine2.simulate_engine()
    # rocket_engine2.graph_rocket(200000, 101325, r=0.02, property='T')

    rocket_engine2.engine_animation(grid_x_points=100, grid_y_points=100, property='T', t_points=100, tf_percentage=1, save=True)




