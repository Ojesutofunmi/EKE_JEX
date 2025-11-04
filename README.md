# EKE Analysis - README

## Overview
This repository contains the setup and processing pipeline for Eddy Kinetic Energy (EKE) analysis using Jexpresso simulations with post-processing in Python. The analysis workflow is documented in `EKE_simone.ipynb`.

## Case 1: Cold Microphysics - SAM

### Directory Structure
Navigate to the `CompEuler` folder and locate the following configuration directories:
- `coarse_mp=sam_config5`
- `medium_mp=sam_config5`
- `fine_mp=sam_config5`

**Note:** Only `config5` configurations are used for this analysis.

### Grid Configuration

All grid files have been generated using GMSH and are available in the `grid_meshes` folder.

| Grid Type | Domain Size (km) | Δx (m) | Δz (m) | Order 4 | Order 5 | Order 6 | WRF Grid Points | Grid File Name (Order 4) | Grid File Name (Order 5) | Grid File Name (Order 6) |
|-----------|------------------|--------|--------|---------|---------|---------|-----------------|--------------------------|--------------------------|--------------------------|
| Fine | 150 × 24 | 200 | 200 | 188 × 30 | 150 × 24 | 125 × 20 | 751 × 121 | `fine_grid_4.msh` | `fine_grid_5.msh` | `fine_grid_6.msh` |
| 400m | 150 × 24 | 400 | 200 | 94 × 30 | 75 × 24 | 63 × 20 | 376 × 121 | `fine_400_4.msh` | `fine_400_5.msh` | `fine_400_6.msh` |
| 600m | 150 × 24 | 600 | 300 | 63 × 20 | 50 × 16 | 42 × 14 | 251 × 81 | `fine_600_4.msh` | `fine_600_5.msh` | `fine_600_6.msh` |
| 800m | 150 × 24 | 800 | 300 | 47 × 20 | 38 × 16 | 32 × 14 | 188 × 81 | `fine_800_4.msh` | `fine_800_5.msh` | `fine_800_6.msh` |
| Medium | 150 × 24 | 1,200 | 400 | 31 × 15 | 25 × 12 | 21 × 10 | 126 × 61 | `medium_4.msh` | `medium_5.msh` | `medium_6.msh` |
| Coarse | 150 × 24 | 4,200 | 400 | 9 × 15 | 7 × 12 | 6 × 10 | 37 × 61 | `coarse_4.msh` | `coarse_5.msh` | `coarse_6.msh` |

### Grid File Naming Convention

- **Fine grid (200m):** `fine_grid_{order}.msh`
- **400m, 600m, 800m grids:** `fine_{resolution}_{order}.msh`
- **Medium and Coarse grids:** `{grid_type}_{order}.msh`

where `{order}` is 4, 5, or 6.

### Sounding File
All simulations use the sounding file: `test_sounding.data`

## Case 2: Cold Microphysics - Kessler

### Overview
This case is identical to the SAM case except for the microphysics scheme. The SAM code has been modified to implement Kessler microphysics only, which has also been implemented in WRF.

**Important:** Two separate Jexpresso model instances are used - one strictly for SAM and another for Kessler - to prevent interference between the schemes.

### Directory Structure
Navigate to the `CompEuler` folder and locate the following configuration directories:
- `coarse_mp=k_config5`
- `medium_mp=k_config5`
- `fine_mp=k_config5`

**Note:** Only `config5` configurations are used for this analysis.

### Microphysics File
The modified Kessler microphysics file is included in this repository (see attached files).

### Grid and Sounding Files
All grid files and the sounding file remain the same as in Case 1 (SAM).

## Configuration 5: High Dissipation Regime

The following configuration parameters are used for the high dissipation regime simulations:

| Config | Purpose | WRF khdif | WRF kvdif | WRF damp_opt | WRF dampcoef | WRF zdamp | JExpresso μ | JExpresso ivisc_equations | JExpresso ctop_mult | Expected Outcome |
|--------|---------|-----------|-----------|--------------|--------------|-----------|-------------|---------------------------|---------------------|------------------|
| **5** | High dissipation regime | 150 | 150 | 2 | 0.01 | 16000 | [0,150,150,150,150,150] | [2,3,4,5,6] | 0.01 | Upper limit: convection survival test |

### Sponge Layer Implementation

**Important Note:** The sponge layer coefficient in Jexpresso is controlled by the `ctop_mult` parameter. This implementation has been modified in the `user_source.jl` file to match the WRF sponge layer behavior specified by `damp_opt`, `dampcoef`, and `zdamp` parameters.

## Running the Analysis

1. Ensure all grid files from `grid_meshes` folder are in the appropriate locations
2. Place `test_sounding.data` in the working directory
3. Select the appropriate configuration folder based on your case (SAM or Kessler) and resolution
4. Configure parameters according to Configuration 5 table above
5. Verify that `user_source.jl` contains the modified sponge layer implementation
6. Run the Jexpresso simulation
7. Use `EKE_simone.ipynb` for post-processing and EKE plot generation

## Files Included

- `EKE_simone.ipynb` - Python notebook for EKE post-processing and visualization
- `grid_meshes/` - Folder containing all GMSH-generated grid files
- `test_sounding.data` - Atmospheric sounding input file
- `user_source.jl` - Modified source file with sponge layer implementation
- Modified Kessler microphysics file (for Case 2)
