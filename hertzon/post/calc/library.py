"""Ready-made expressions and functions of the Fields Calculator: copied into a design and changed there."""
from __future__ import annotations

# (name, expression, description): field densities, used in plots or integrated with int_vol / int_surf
EXPRESSIONS = [
    ("p_diel", "0.5*omega*eps0*eps_r*tand*|E|^2", "Dielectric loss density (W/m³), time average"),
    ("p_ohm", "0.5*sigma*|E|^2", "Ohmic loss density (W/m³) in conductors that are solved inside"),
    ("w_e", "0.25*eps0*eps_r*|E|^2", "Electric energy density (J/m³), time average"),
    ("w_m", "0.25*mu0*mu_r*|H|^2", "Magnetic energy density (J/m³), time average"),
    ("S_active", "re(S)", "Active power density, the real part of the Poynting vector (W/m²)"),
    ("p_wall_cu", "0.5*rs(5.8e7S/m)*|Js|^2", "Wall loss density of a copper surface (W/m²), faces only"),
    ("E_rms", "|rms(E)|", "RMS magnitude of E (V/m)"),
    ("E_inst", "inst(E)", "E at the phase of the context (V/m), instantaneous"),
    ("maxwell_residual", "|curl(E) + j*omega*mu0*mu_r*H|/(omega*mu0*|H|)",
     "Check of the solution: relative residual of Faraday's law"),
]

# functions "name(params) = body": geometries are passed as arguments
FUNCTIONS = [
    ("voltage(L) = int_line(L, E)", "Voltage along a line (V), direction of the line"),
    ("current(C) = circ(C, H)", "Current through a closed line (A), right-hand rule"),
    ("power(A) = flux(A, re(S))", "Active power through a face (W), direction of the normal"),
    ("loss_diel(V) = int_vol(V, 0.5*omega*eps0*eps_r*tand*|E|^2)", "Dielectric loss in a volume (W)"),
    ("loss_wall(A, sig) = int_surf(A, 0.5*rs(sig)*|Js|^2)", "Wall loss of a face for a conductivity sig (W)"),
    ("energy_e(V) = int_vol(V, 0.25*eps0*eps_r*|E|^2)", "Electric energy in a volume (J), time average"),
    ("energy_m(V) = int_vol(V, 0.25*mu0*mu_r*|H|^2)", "Magnetic energy in a volume (J), time average"),
    ("q_diel(V) = omega*(energy_e(V) + energy_m(V))/loss_diel(V)", "Quality factor from the dielectric loss"),
    ("emax(G) = max(G, |E|)", "Peak field strength on a geometry (V/m)"),
]
