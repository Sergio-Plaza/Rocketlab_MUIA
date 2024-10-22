import numpy as np
import scipy.optimize
from typing import Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass

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

    def engine_constants(self):
        """
        Calculates pressure ratios and mach numbers for chocked flow in the subsonic and supersonic case. 
        Calulates the pressure ratio with a nomal shock wave at the exit #! Revisar esto de las ondas de choque
        """
        f = lambda X: self.e - self.gammadegamma / (X ** (1 / self.gamma) * (2 * self.gamma / (self.gamma - 1) * (1 - X ** ((self.gamma - 1) / self.gamma))) ** 0.5)
        self.sub = scipy.optimize.fsolve(f, 0.9)[0]  # Ps/Pc exterior chamber pressure ratio with chocked flow for subsonic case
        self.sup = scipy.optimize.fsolve(f, 0.1)[0]  # Ps/Pc exterior chamber pressure ratio with chocked flow for supersonic case

        self.msup = np.sqrt(2 / (self.gamma - 1) * (1 - self.sup ** ((self.gamma - 1) / self.gamma)) / (self.sup ** ((self.gamma - 1) / self.gamma)))  # Mach number in the supersonic case
        self.msub = np.sqrt(2 / (self.gamma - 1) * (1 - self.sub ** ((self.gamma - 1) / self.gamma)) / (self.sub ** ((self.gamma - 1) / self.gamma)))  # Mach number in the subsonic case

        m1n = self.msup
        m2n = np.sqrt((2 + (self.gamma - 1) * m1n ** 2) / (2 * self.gamma * m1n ** 2 - (self.gamma - 1)))
        p02_entre_pa = (1 + (self.gamma - 1) / 2 * m2n ** 2) ** (self.gamma / (self.gamma - 1))
        salto_P0_OCN = (((2 * self.gamma * m1n ** 2 - (self.gamma - 1)) / (self.gamma + 1)) * ((2 + (self.gamma - 1) * m1n ** 2) / ((self.gamma + 1) * m1n ** 2)) ** self.gamma) ** (1 / (self.gamma - 1))
        self.salto = p02_entre_pa * salto_P0_OCN
    
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

        self.Pc_t.append(Pc_new)
        self.r_t.append(r_new)

        # Calculates and updates engine properties
        self.E_t.append(self.calculate_thrust(Pc_new, Pa))
        self.Mp_t.append(self.calculate_propulsant(r_new))
    
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
            Ms = np.sqrt(2 / (self.gamma -1) * np.abs(((Pc / Pa) ** ((self.gamma -1) / self.gamma) - 1)))
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
            _type_: _description_
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

class rocket():
    class SimulationResults:
        def __init__(self, results):
            self.results = results

        def __str__(self):
            results_str = "\n".join([f"{key}: {value:.3f}" for key, value in self.results.items()])
            return f"Simulation Results:\n{results_str}"
        
    def __init__(self, parameters):
        self.set_parameters(parameters) # Set all parameters as class attributes dynamically

        # Calculate mass and geometry properties based on initial parameters
        self.calculate_mass_properties()
        self.S = np.pi * (self.R + self.t_chamber) ** 2 

        dt_min = min(self.dt_engine, self.dt_trayectory)   # Smallest time step is limitant for both problems until engine shut down

        self.initialize_trajectory(dt_min)                 # Initialize trajectory module
        self.initialize_engine(dt_min)                     # Initialize engine module

        self.simulation()                                  # Starts a simulation

    def set_parameters(self, parameters):
        """Set all parameters as class attributes dynamically."""
        for key, value in parameters.items():
            setattr(self, key, value)
    
    def calculate_mass_properties(self):
        Mc = np.pi * ((self.R + self.t_chamber) ** 2 - self.R ** 2) * self.L * self.rho_c                   # Mass of the chamber
        Mcone = np.pi * self.R * self.R/ np.sin(np.deg2rad(self.alpha)) * self.t_cone * self.rho_cone       # Mass of the cone
        Mp0 = np.pi * (self.R ** 2 - self.R0 ** 2) * self.L * self.rho_pr                                   # Inital fuel mass

        self.Mi = Mc + Mcone + self.Mpl                                                                     # Inert mass (all without fuel)
        self.M0 = self.Mi + Mp0                                                                             # Total inital mass

    def initialize_trajectory(self, dt_min):
        """Initialize the trajectory module with the necessary configurations."""
        trayectory_geometry_parameters = TrayectoryGeometryParameters(M0=self.M0, S=self.S)
        trayectory_physical_constants = TrayectoryPhysicalConstants(g0=self.g0, Ra=self.Ra, Re=self.Re)
        trayectory_initial_conditions = TrayectoryInitialConditions(t0=self.t0, h0=self.h0, v0=self.v0)
        trayectory_simulation_configuration = TrayectorySimulationConfiguration(dt=dt_min, solver=self.solver_trayectory)
        
        self.rocket_trayectory = trayectory(
            trayectory_physical_constants,
            trayectory_initial_conditions,
            trayectory_geometry_parameters,
            trayectory_simulation_configuration
        )

        self.Pa0 = self.rocket_trayectory.Pa_t[-1]  # Initial atmospheric preassure for h0

    def initialize_engine(self, dt_min):
        """Initialize the engine module with the necessary configurations."""
        engine_geometry = EngineGeometryParameters(R=self.R, Rg=self.Rg, Rs=self.Rs, R0=self.R0, L=self.L)
        fuel_parameters = EngineFuelParameters(
            Tc=self.Tc, a=self.a, n=self.n, M_molar=self.M_molar, gamma=self.gamma,
            rho_pr=self.rho_pr, Rend=self.Rend
        )
        initial_conditions = EngineInitialConditions(t0=self.t0, Pa0=self.Pa0)
        simulation_configurations = EngineSimulationConfiguration(dt=dt_min, solver=self.solver_engine)
        
        self.rocket_engine = RocketEngine(
            engine_geometry,
            fuel_parameters,
            initial_conditions,
            simulation_configurations
        )

    def simulation(self):
        """
        Performs a simulation of the rocket coupling the engine performance with the trayectory through the atmospheric preassure
        and the instaneous mass and thrust of the rocket. Both simulations are performed wiht the lowest step and when the engine 
        stops the trayectory step can be set to a higher value.
        """
        while True:
            self.rocket_engine.solve_step(Pa=self.rocket_trayectory.Pa_t[-1])
            self.rocket_trayectory.solve_step(dt=self.dt_trayectory if self.rocket_engine.flag_stop_condition('engine_stop') else None, 
                                              E=self.rocket_engine.E_t[-1], Cd=0.5, M=self.rocket_engine.Mp_t[-1]+self.Mi)
            if self.rocket_trayectory.flag_stop_condition(self.stop_condition): break 

        self.results()

    def graphs(self):
        """
        Graphs the performance characteristics of the rocket.
        """
        self.rocket_engine.graphs()
        self.rocket_trayectory.graphs()

    def stress_analysis(self):
        """
        Calculates the circumferential stress in the chamber due to the gas presure in the worst case (Pa=0) and aplying a safety factor of 1.5.

        Returns:
            float: circumferential stress in MPa.
        """
        SF = 1.5
        Pc_max = np.max(self.rocket_engine.Pc_t)
        sigma_r = Pc_max * (self.R/self.t_chamber) * SF
        return sigma_r / 10 ** 6  #MPa
    
    def results(self):
        """
        Calculates some of the key performance parameters for optimization purposes.

        Returns:
            float, float, float: maximum heigth of the trayectory, total initial mass and circumferential stress in the chamber.
        """
        h_max = np.max(self.rocket_trayectory.h_t)
        M_total = self.M0
        sigma_r = self.stress_analysis()
        
        results = {'h_max': h_max, 'M_total': M_total, 'sigma_r': sigma_r}
        self.simulation_results = self.SimulationResults(results)
        return results

    def __str__(self):
        results_str = "\n".join([f"{key}: {value:.3f}" for key, value in self.simulation_results.items()])
        return f"RocketSimulation Results:\n{results_str}"
    


