#-------------------------------------------------------------------------------------------
# Microphysics (mp) variables:
#-------------------------------------------------------------------------------------------
Base.@kwdef mutable struct St_Microphysics{T <: AbstractFloat, dims1, dims2, dims3, backend}
    Tabs    = KernelAbstractions.zeros(backend,  T, dims1)
    qn      = KernelAbstractions.zeros(backend,  T, dims1)
    qc      = KernelAbstractions.zeros(backend,  T, dims1)
    qi      = KernelAbstractions.zeros(backend,  T, dims1)
    qr      = KernelAbstractions.zeros(backend,  T, dims1)
    qs      = KernelAbstractions.zeros(backend,  T, dims1)
    qg      = KernelAbstractions.zeros(backend,  T, dims1)
    Pr      = KernelAbstractions.zeros(backend,  T, dims1)
    Ps      = KernelAbstractions.zeros(backend,  T, dims1)
    Pg      = KernelAbstractions.zeros(backend,  T, dims1)
    S_micro = KernelAbstractions.zeros(backend,  T, dims1)
    rainnc  = KernelAbstractions.zeros(backend,  T, dims1)
    rainncv = KernelAbstractions.zeros(backend,  T, dims1)
    vt      = KernelAbstractions.zeros(backend,  T, dims1)
    prod    = KernelAbstractions.zeros(backend,  T, dims1)
    prodk   = KernelAbstractions.zeros(backend,  T, dims1)
    vtden   = KernelAbstractions.zeros(backend,  T, dims1)
    rdzk    = KernelAbstractions.zeros(backend,  T, dims1)
    rdzw    = KernelAbstractions.zeros(backend,  T, dims1)
    ρk      = KernelAbstractions.zeros(backend,  T, dims1)
    temp1   = KernelAbstractions.zeros(backend,  T, dims1)
    temp2   = KernelAbstractions.zeros(backend,  T, dims1)
    qsatt   = KernelAbstractions.zeros(backend,  T, dims1)
    # For variables needing 3D allocation:
    dhldt   = KernelAbstractions.zeros(backend,  T, dims3)
    dqtdt   = KernelAbstractions.zeros(backend,  T, dims3)
    dqpdt   = KernelAbstractions.zeros(backend,  T, dims3)
    drad_sw = KernelAbstractions.zeros(backend,  T, dims3)
    drad_lw = KernelAbstractions.zeros(backend,  T, dims3)
    flux_lw = KernelAbstractions.zeros(backend,  T, dims1)
    flux_sw = KernelAbstractions.zeros(backend,  T, dims1)
end

function allocate_Microphysics(nelem, npoin, ngl, T, backend, SD; lmoist=false)
    if lmoist
        if SD == NSD_3D()
            dims1 = (Int64(npoin))
            dims2 = (Int64(nelem))
            dims3 = (Int64(nelem), Int64(ngl), Int64(ngl), Int64(ngl))
        else
            dims1 = (Int64(npoin))
            dims2 = (Int64(nelem))
            dims3 = (Int64(nelem), Int64(ngl), Int64(ngl))
        end
    else
        dims1 = (Int64(1))
        dims2 = (Int64(1))
        dims3 = (Int64(1))
    end

    mp = St_Microphysics{T, dims1, dims2, dims3, backend}()
    return mp
end

Base.@kwdef mutable struct St_SamMicrophysics{T <:AbstractFloat, dims1, dims2, dims3, backend}

    Tabs    = KernelAbstractions.zeros(backend,  T, dims1) #Absolute temperature
    qn      = KernelAbstractions.zeros(backend,  T, dims1) #total cloud 
    qi      = KernelAbstractions.zeros(backend,  T, dims1) #ice cloud
    qc      = KernelAbstractions.zeros(backend,  T, dims1) #cloud water
    qr      = KernelAbstractions.zeros(backend,  T, dims1) #rain
    qs      = KernelAbstractions.zeros(backend,  T, dims1) #snow
    qg      = KernelAbstractions.zeros(backend,  T, dims1) #graupel
    Pr      = KernelAbstractions.zeros(backend,  T, dims1) #rain precipitation flux
    Ps      = KernelAbstractions.zeros(backend,  T, dims1) #snow precipitation flux
    Pg      = KernelAbstractions.zeros(backend,  T, dims1) #graupel precipitation flux
    S_micro = KernelAbstractions.zeros(backend,  T, dims1)  #microphysical source term
    qsatt   = KernelAbstractions.zeros(backend,  T, dims1)  #saturation vapor fraction
    dhldt   = KernelAbstractions.zeros(backend,  T, dims3, ) #Storage for preciptation source contributions to hl
    dqtdt   = KernelAbstractions.zeros(backend,  T, dims3) #Storage preciptation source contributions to qt
    dqpdt   = KernelAbstractions.zeros(backend,  T, dims3) #Storage preciptation source contributions to qp
    drad_sw   = KernelAbstractions.zeros(backend,  T, dims3) #Storage longwave flux contributions
    drad_lw   = KernelAbstractions.zeros(backend,  T, dims3) #Storage shortwave flux contribution
    flux_lw = KernelAbstractions.zeros(backend,  T, dims1) # storage for longwave flux
    flux_sw = KernelAbstractions.zeros(backend,  T, dims1) # storage for shortwave flux
end

function allocate_SamMicrophysics(nelem, npoin, ngl, T, backend , SD; lmoist=false)

    if lmoist
        if (SD == NSD_3D())
            dims1 = (Int64(npoin))
            dims2 = (Int64(nelem))
            dims3 = (Int64(nelem), Int64(ngl), Int64(ngl), Int64(ngl))
        else
            dims1 = (Int64(npoin))
            dims2 = (Int64(nelem))
            dims3 = (Int64(nelem), Int64(ngl), Int64(ngl))
        end

    else
        dims1 = (Int64(1))
        dims2 = (Int64(1))
        dims3 = (Int64(1)) 
    end

    mp = St_SamMicrophysics{T, dims1, dims2, dims3, backend}()

    return mp
end
