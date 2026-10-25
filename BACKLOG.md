# Backlog

Small improvements collected for a future release. Items move to `CHANGELOG.md` once they are done.

## Open

- **Ports cut by a symmetry plane**: apply the `ImpedanceMultiplier` of the symmetry boundary to the port
  impedance. Today a lumped port cut by the plane is referred to the impedance of the half port (half
  dipole: S11 0.31 instead of 0.06); far field pattern and directivity are not affected.
