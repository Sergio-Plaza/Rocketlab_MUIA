import numpy as np 
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class AerodynamicsGeometryParameters:
    """Geometry parameters of the rocket."""
    D: float                                     # Cylinder (body) diameter in meters
    alpha_cone: float                            # Angle of the cone in degrees
    L_body: float                                # Length of the chamber in meters
    Rs: float                                    # Exit radius of the nozzle in meters

    def __post_init__(self):
        # Ensure all dimensions are positive
        if any(dimension <= 0 for dimension in [self.D, self.alpha_cone, self.L_body]): 
            raise ValueError("All dimensions must be positive numbers.")

        # Calculate geometric parameters of the rocket
        self.L_cone = self.D / (2 * np.tan(np.deg2rad(self.alpha_cone)))                            # Length of the cone
        self.L = self.L_cone + self.L_body                                                          # Total length of the rocket
        self.S_wet_body = np.pi * self.D * self.L_body                                              # Wetted surface of the body
        self.S_wet_cone = np.pi * self.D * self.L_cone / (2 * np.sin(np.deg2rad(self.alpha_cone)))  # Wetted surface of the cone
        self.S = np.pi * self.D ** 2 / 4                                                            # Reference surface
        self.As = np.pi * self.Rs ** 2                                                              # Exit area of the nozzle

@dataclass       
class AerodynamicsFlowConstants:
    """Aerodynamic flow constants."""
    mu: float                                       # Air viscosity coefficient in Pa·s
    gamma: float                                    # Specific heat ratio of air
    M_molar: float                                  # Air molar mass in kg/mol

    def __post_init__(self):
        # Ensure all constants are positive
        if any(dimension <= 0 for dimension in [self.mu, self.gamma, self.M_molar]): 
            raise ValueError("All constants must be positive numbers.")

        self.Rgas = 8.314 / self.M_molar            # Specific gas constant for air in J/(kg·K)
class aerodynamics:
    def __init__(self, geometry: AerodynamicsGeometryParameters, flow_constants: AerodynamicsFlowConstants):
        # Initialize class by updating with geometry and flow constant attributes
        self.__dict__.update(geometry.__dict__)
        self.__dict__.update(flow_constants.__dict__)

    def calculate_Mach(self, V: float, T: float):
        """Calculate Mach number based on velocity V and temperature T.

        Args:
            V (float): Flow velocity in m/s
            T (float): Flow temperature in K

        Returns:
            float: Mach number of the flow
        """
        return V / np.sqrt(self.gamma * self.Rgas * T)
    
    ## Cd0 Frontal 
    def calculate_cd0_frontal(self, M: float, cd0_f: float):
        """Calculate the frontal drag coefficient (Cd0) for various regimes.

        Args:
            M (float): Mach number of the air flow upstream
            Cd0_f (float): Friction contribution of the drag coefficient

        Returns:
            float: Frontal drag coefficient based on the Mach regime
        """
        if M < 0.8: # Subsonic Regime
            return self.cd0_frontal_sub(cd0_f)
        if M > 1.2: # Supersonic Regime
            return self.cd0_frontal_sup(M)
        else:       # Transonic Regime
            return self.cd0_frontal_trans(M, cd0_f)

    def cd0_frontal_sub(self, cd0_f_body: float):
        """Calculate the frontal drag coefficient in the subsonic regime.

        Args:
            Cd0_f_body (float): Friction contribution of the drag coefficient

        Returns:
            float: Frontal drag coefficient for subsonic flow
        """
        return (60 / (self.L / self.D) ** 3 + 0.0025 * self.L / self.D) * cd0_f_body # Ec 7-4 REF 1

    def cd0_frontal_trans(self, M: float, cd0_f: float):
        """Calculate the frontal drag coefficient in the transonic regime.

        Args:
            M (float): Mach number of the air flow upstream
            Cd0_f (float): Friction contribution of the drag coefficient

        Returns:
            float: Frontal drag coefficient for transonic flow
        """
        weight_sub = (1.2 - M) / (1.2 - 0.8)
        weight_sup = (M - 0.8) / (1.2 - 0.8)
        return weight_sub * self.cd0_frontal_sub(cd0_f) + weight_sup * self.cd0_frontal_sup(M)

    def cd0_frontal_sup(self, M: float) -> float:
        """Calculate the frontal drag coefficient in the supersonic regime.

        Args:
            M (float): Mach number of the air flow upstream

        Returns:
            float: Frontal drag coefficient for supersonic flow
        """
        return (0.0083 + 0.0096 / M ** 2) * (self.alpha_cone / 10) ** 1.69 # Ec 7-5 REF 1
    
    ## Cd0 de Base
    def calculate_cd0_base(self, M: float, engine_on: bool):
        """Calculate the base drag coefficient (Cd0) for various regimes.

        Args:
            M (float): Mach number of the air flow upstream
            engine_on (bool): Indicates if the engine is on (affects base drag)

        Returns:
            float: Base drag coefficient based on the Mach regime
        """
        if M < 0.8: # Subsonic Regime
            return self.cd0_base_sub(M, engine_on)
        if M > 1.2: # Supersonic Regime
            return self.cd0_base_sup(M, engine_on)
        else:       # Transonic Regime
            return self.cd0_base_trans(M, engine_on)
        
    def cd0_base_sub(self, M: float, engine_on: bool):
        """Calculate the base drag coefficient in the subsonic regime.

        Args:
            M (float): Mach number of the air flow upstream
            engine_on (bool): Indicates if the engine is on (affects base drag)

        Returns:
            float: Base drag coefficient for subsonic flow
        """
        if engine_on:
            return (0.12 + 0.13 * M ** 2) * (self.S - self.As) / self.S # Ec 7.12 REF 1
        else:
            return (0.12 + 0.13 * M ** 2)                               # Ec 7.10 REF 1
    
    def cd0_base_trans(self, M: float, engine_on: bool):
        """Calculate the base drag coefficient in the transonic regime.

        Args:
            M (float): Mach number of the air flow upstream
            engine_on (bool): Indicates if the engine is on (affects base drag)

        Returns:
            float: Base drag coefficient for transonic flow
        """
        weight_sub = (1.2 - M) / (1.2 - 0.8)
        weight_sup = (M - 0.8) / (1.2 - 0.8)
        return weight_sub * self.cd0_base_sub(M, engine_on) + weight_sup * self.cd0_base_sup(M, engine_on)
    
    def cd0_base_sup(self, M: float, engine_on: bool):
        """Calculate the base drag coefficient in the supersonic regime.

        Args:
            M (float): Mach number of the air flow upstream
            engine_on (bool): Indicates if the engine is on (affects base drag)

        Returns:
            float: Base drag coefficient for supersonic flow
        """
        if engine_on:
            return 0.25 / M ** 2 * (self.S - self.As) / self.S  # Ec 7-12 REF 1
        else:
            return 0.25 / M ** 2                                # Ec 7-11 REF 1

    ## Cd0 de Fricción
    def calculate_cd0_f(self, M: float, V: float, rho: float):
        """Calculate the friction drag coefficient for the rocket body.

        Args:
            M (float): Mach number of the air flow upstream
            V (float): Flow velocity in m/s
            rho (float): Air density in kg/m^3

        Returns:
            float: Friction drag coefficient for the rocket
        """
        K_cone = 4 / np.sqrt(3)
        K_body = 1.28 
        K = [K_cone, K_body]
        L = [self.L_cone, self.L_body]
        S_wet = [self.S_wet_cone, self.S_wet_body]
        Cfl_list = []
        
        for i in range(2):
            Re = rho * V * L[i] / self.mu                                           # Calculate Reynolds number EC 7-15 REF 1

            if Re < 10 ** 6:               # Laminar case
                Cfi = 0.664 / np.sqrt(Re) if Re > 1 else 0                          # Ec 7-16 REF 1
                Cfl = Cfi * (1 / (1 + 0.17 * M ** 2)) ** 0.1295 if M < 1 else Cfi   # Ec 7-18 REF 1

            else:                          # Turbulent case
                Cfi = 0.288 / (np.log10(Re) ** 2.45)                                # Ec 7-17 REF 1
                Cfl = Cfi / (1 + (self.gamma - 1) / 2 * M ** 2) ** 0.467 if M > 1 else Cfi / (1 + 0.008 * M ** 2) # Ec 7-19 and 7-20 REF 1
            
            Cfl_list.append(Cfl * K[i] * S_wet[i] / self.S)  # Friction coefficient contribution Ec 7-24 REF 1

        return sum(Cfl_list)

    def calculate_Cd0(self, V: float, rho: float, T: float, engine_on: bool):
        """Calculate the total drag coefficient (Cd0) for the rocket.

        Args:
            V (float): Flow velocity in m/s
            rho (float): Air density in kg/m^3
            T (float): Air temperature in K
            engine_on (bool): Indicates if the engine is on (affects base drag)

        Returns:
            float: Total drag coefficient (Cd0), including friction, frontal, and base drag contributions
        """
        M = self.calculate_Mach(V, T)

        cd0_f = self.calculate_cd0_f(M, V, rho)
        cd0_frontal = self.calculate_cd0_frontal(M, cd0_f)
        cd0_base = self.calculate_cd0_base(M, engine_on)

        return cd0_frontal + cd0_base + cd0_f

    def plot_cd0_contributions(self, V_lim: tuple, T: float, rho: float, engine_on: bool):
        """Plot Cd0 contributions from friction, frontal, and base drag over a range of velocities.

        Args:
            V_lim (tuple): Velocity range (start, end) in m/s for the plot
            T (float): Air temperature in K
            rho (float): Air density in kg/m^3
            engine_on (bool): Indicates if the engine is on (affects base drag)
        """
        M_values = []
        total_cd0 = []
        cd0_f_values = []
        cd0_frontal_values = []
        cd0_base_values = []

        V_range = np.linspace(V_lim[0], V_lim[1])
        for V in V_range:
            M = self.calculate_Mach(V, T)
            Cd0_f = self.calculate_cd0_f(M, V, rho)
            Cd0_frontal = self.calculate_cd0_frontal(M, Cd0_f)
            Cd0_base = self.calculate_cd0_base(M, engine_on)

            M_values.append(M)
            total_cd0.append(Cd0_frontal + Cd0_base + Cd0_f)
            cd0_f_values.append(Cd0_f)
            cd0_frontal_values.append(Cd0_frontal)
            cd0_base_values.append(Cd0_base)

        # Plot contributions and total Cd0
        plt.figure(figsize=(10, 6))
        plt.plot(M_values, total_cd0, label="Cd0 Total", color="black", linewidth=2)
        plt.plot(M_values, cd0_f_values, label="Cd0 Fricción", color="blue", linestyle="--")
        plt.plot(M_values, cd0_frontal_values, label="Cd0 Frontal", color="red", linestyle="--")
        plt.plot(M_values, cd0_base_values, label="Cd0 Base", color="green", linestyle="--")
        
        plt.xlabel("Velocidad (m/s)")
        plt.ylabel("Coeficiente de Arrastre (Cd0)")
        plt.title("Contribuciones al Cd0 en función de la velocidad")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    # Crear instancias de las dataclass
    aerodynamics_geometry_parameters = AerodynamicsGeometryParameters(D=5.25e-02,  alpha_cone=50, L_body=100e-02, Rs=1e-02)
    aerodynamcis_flow_constants = AerodynamicsFlowConstants(M_molar=28.97e-3,  gamma=1.4, mu=1.82e-05)

    # Instanciar la clase aerodynamics
    aerodynamics1 = aerodynamics(aerodynamics_geometry_parameters, aerodynamcis_flow_constants)
    
    # Calcular y graficar Cd0 con contribuciones
    aerodynamics1.calculate_Cd0(V=0, rho=1.225, T=300, engine_on=False) 
    aerodynamics1.plot_cd0_contributions([0, 800], rho=1.225, T=300, engine_on=False)