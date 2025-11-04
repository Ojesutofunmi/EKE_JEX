@kernel function do_micro_physics_gpu_3D!(uaux, qe, Tabs, qn, qi, qc, qr, qs, qg, Pr, Ps, Pg, S_micro, PhysConst, MicroConst, lpert, neq, npoin, z, adjusted, Pm)
    ip = @index(Global, Linear)
    T = eltype(uaux)
    if (lpert)
        @inbounds uaux[ip,6] = max(zero(T), (uaux[ip,6] + qe[ip,6])) - qe[ip,6]
        @inbounds uaux[ip,7] = max(zero(T), (uaux[ip,7] + qe[ip,7])) - qe[ip,7]
    else
        @inbounds uaux[ip,6] = max(zero(T), uaux[ip,6])
        @inbounds uaux[ip,7] = max(zero(T), uaux[ip,7])
    end
    uip = @view(uaux[ip,1:neq])
    qeip = @view(qe[ip,1:neq+1])

    # Only Kessler: no snow/graupel/ice
    @inbounds adjusted[ip,:] .= saturation_adjustment_sam_microphysics_gpu(uip, qeip, z[ip], MicroConst, PhysConst, lpert)
    @inbounds Tabs[ip] = adjusted[ip,1]
    @inbounds uaux[ip,end] = adjusted[ip,2]
    @inbounds qn[ip] = adjusted[ip,3]
    @inbounds qc[ip] = adjusted[ip,4]
    @inbounds qi[ip] = 0.0
    @inbounds qr[ip] = adjusted[ip,6]
    @inbounds qs[ip] = 0.0
    @inbounds qg[ip] = 0.0
    @inbounds qsatt = adjusted[ip,9]

    @inbounds Pr[ip], Ps[ip], Pg[ip] = compute_Pm_gpu(uip, qeip, qr[ip], MicroConst, lpert)
    S = compute_dqpdt_sam_micro_gpu(uip, qeip, Tabs[ip], qn[ip], qc[ip], qr[ip], qsatt, MicroConst, PhysConst, lpert)
    @inbounds S_micro[ip] = S
end


function do_micro_physics!(Tabs, qn, qc, qi, qr, qs, qg, Pr, Ps, Pg, S_micro, qsatt, npoin, uaux, z, qe, NSD, ::PERT)
    PhysConst = PhysicalConst{Float64}()
    MicroConst = MicrophysicalConst{Float64}()

    saturation_adjustment_sam_microphysics!(uaux, qe, Tabs, qn, qi, qc, qr, qs, qg, qsatt, z, npoin, MicroConst, PhysConst, NSD, true)
    compute_Pm!(uaux, qe, qr, qs, qg, Pr, Ps, Pg, npoin, MicroConst, true)
    compute_dqpdt_sam_micro!(uaux, qe, Tabs, qn, qc, qi, qr, qs, qg, qsatt, S_micro, npoin, MicroConst, PhysConst, NSD, true)

    # Set non-Kessler fields to zero
    qi .= 0.0; qs .= 0.0; qg .= 0.0; Ps .= 0.0; Pg .= 0.0
    return nothing
end

function do_micro_physics!(Tabs, qn, qc, qi, qr, qs, qg, Pr, Ps, Pg, S_micro, qsatt, npoin, uaux, z, qe, NSD, ::TOTAL)
    PhysConst = PhysicalConst{Float64}()
    MicroConst = MicrophysicalConst{Float64}()

    saturation_adjustment_sam_microphysics!(uaux, qe, Tabs, qn, qi, qc, qr, qs, qg, qsatt, z, npoin, MicroConst, PhysConst, NSD, false)
    compute_Pm!(uaux, qe, qr, qs, qg, Pr, Ps, Pg, npoin, MicroConst, false)
    compute_dqpdt_sam_micro!(uaux, qe, Tabs, qn, qc, qi, qr, qs, qg, qsatt, S_micro, npoin, MicroConst, PhysConst, NSD, false)

    # Set non-Kessler fields to zero
    qi .= 0.0; qs .= 0.0; qg .= 0.0; Ps .= 0.0; Pg .= 0.0
    return nothing
end

function compute_precipitation_derivatives!(dqpdt, dqtdt, dhldt, Pr, Ps, Pg, Tabs, qi, ρ, ρe, nelem, ngl, connijk, H, metrics, ω, dψ, SD::NSD_2D, ::TOTAL)
    MicroConst = MicrophysicalConst{Float64}()
    dqpdt .= 0.0
    dqtdt .= 0.0
    dhldt .= 0.0
    Lc = MicroConst.Lc

    # For Kessler, snow/graupel/ice terms are ignored
    for e=1:nelem
        # Only rain is active; Ps and Pg are zero
        for i=1:ngl, j=1:ngl
            ip = connijk[e,i,j]
            H[i,j,1] = Pr[ip]
        end
        compute_vertical_derivative_q!(dqpdt, H, e, ngl, metrics.Je, metrics.dξdy, metrics.dηdy, ω, dψ, SD)

        # Only Lc*Pr[ip]; Ls*(Ps+Pg) is zero
        for i=1:ngl, j=1:ngl
            ip = connijk[e,i,j]
            H[i,j,1] = Lc * Pr[ip]
        end
        compute_vertical_derivative_q!(dhldt, H, e, ngl, metrics.Je, metrics.dξdy, metrics.dηdy, ω, dψ, SD)

        # No cloud ice sedimentation in Kessler: dqtdt is zero
        for i=1:ngl, j=1:ngl
            H[i,j,1] = 0.0
        end
        compute_vertical_derivative_q!(dqtdt, H, e, ngl, metrics.Je, metrics.dξdy, metrics.dηdy, ω, dψ, SD)

        # The following is only for mixed-phase; set to zero for Kessler
        for i=1:ngl, j=1:ngl
            dhldt[e,i,j] += 0.0
        end
    end
end



function compute_precipitation_derivatives!(dqpdt, dqtdt, dhldt, Pr, Ps, Pg, Tabs, qi, ρ, ρe, nelem, ngl, connijk, H, metrics, ω, dψ, SD::NSD_2D, ::PERT)
    MicroConst = MicrophysicalConst{Float64}()
    dqpdt .= 0.0
    dqtdt .= 0.0
    dhldt .= 0.0
    Lc = MicroConst.Lc

    # Kessler: ignore Ps, Pg, qi, dqtdt, ωn, etc.
    for e=1:nelem
        # Only rain; Ps, Pg are always zero
        for i=1:ngl, j=1:ngl
            ip = connijk[e,i,j]
            H[i,j,1] = Pr[ip]
        end
        compute_vertical_derivative_q!(dqpdt, H, e, ngl, metrics.Je, metrics.dξdy, metrics.dηdy, ω, dψ, SD)

        # Only Lc*Pr[ip]; Ls*(Ps+Pg) is zero
        for i=1:ngl, j=1:ngl
            ip = connijk[e,i,j]
            H[i,j,1] = Lc * Pr[ip]
        end
        compute_vertical_derivative_q!(dhldt, H, e, ngl, metrics.Je, metrics.dξdy, metrics.dηdy, ω, dψ, SD)

        # No ice sedimentation; dqtdt zero
        for i=1:ngl, j=1:ngl
            H[i,j,1] = 0.0
        end
        compute_vertical_derivative_q!(dqtdt, H, e, ngl, metrics.Je, metrics.dξdy, metrics.dηdy, ω, dψ, SD)

        # dhldt update with dqtdt is zero for Kessler
        for i=1:ngl, j=1:ngl
            dhldt[e,i,j] += 0.0
        end
    end
end



function precipitation_flux_gpu(u,qe,MicroConst,lpert,Pr,Ps,Pg,qi)
    T= eltype(u)
    if (lpert)
        ρ = u[1] + qe[1]
    else
        ρ = u[1]
    end
    Lc = MicroConst.Lc
    Ls = MicroConst.Ls

    return T(0.0), T(Lc*Pr), T(0.0), T(Pr)
end

function add_micro_precip_sources!(mp::St_SamMicrophysics,flux_lw, flux_sw, T,S_micro,S,q,qn,qe, ::NSD_2D, ::TOTAL)
    
    PhysConst = PhysicalConst{Float64}()
    
    ρ        = q[1]
    qt       = q[5]/ρ
    qp       = q[6]/ρ
    qv       = qt - qn
    ρqv_pert = ρ*qv - qe[5]

    S[3] += PhysConst.g*(0.61*ρqv_pert -ρ*(qn+qp)) 
    S[4] += ρ*(flux_lw - flux_sw)
    S[5] += -ρ*S_micro
    S[6] += ρ*S_micro

end 
    
function add_micro_precip_sources!(mp::St_SamMicrophysics,flux_lw, flux_sw, T,S_micro,S,q,qn,qe, ::NSD_2D,::PERT)
    
    PhysConst = PhysicalConst{Float64}()
    
    ρ        = q[1]+qe[1]
    qt       = (q[5] + qe[5])/ρ
    qv_ref   = qe[5]/qe[1]
    qp       = (q[6] + qe[6]) /ρ
    qv       = (qt - qn) #- qe[6]/qe[1]
    ρqv_pert = ρ*qv - qe[5]

    S[3] += PhysConst.g*(0.61*ρqv_pert -ρ*(qn+qp))# should we ignore condensates in the hydrostatic balance if they're not included in the pressure term?
    S[4] += ρ*(flux_lw - flux_sw)
    S[5] += -ρ*S_micro
    S[6] += ρ*S_micro

end

function precipitation_source_gpu(u,qe,lpert,qn,S_micro,PhysConst,MicroConst)
    T = eltype(u)
    if (lpert)
        ρ   = u[1] + qe[1]
        qt  = (u[6] + qe[6])/ρ
        qp  = (u[7] + qe[7])/ρ
        ρqv = ρ*(qt - qn) - qe[6]
    else
        ρ   = u[1]
        qt  = u[6]/ρ
        qp  = u[7]/ρ
        ρqv = ρ*(qt - qn)
    end
    
    return T(-PhysConst.g*(T(0.608)*ρqv-ρ*(qn+qp))), T(0.0), T(-ρ*S_micro), T(ρ*S_micro)
end

function compute_dqpdt_sam_micro!(uaux, qe, Tabs, qn, qc, qi, qr, qs, qg, qsatt, S_micro, npoin, MicroConst, PhysConst, NSD, lpert)
    # Kessler: Only rain (no snow, no graupel, no cloud ice)
    a_rain = MicroConst.a_rain
    b_rain = MicroConst.b_rain
    N0_rain = MicroConst.N0_rain
    Er_c = MicroConst.Er_c
    γ3br = MicroConst.γ3br
    ρ_rain = MicroConst.ρ_rain
    C_rain = MicroConst.C_rain
    a_fr   = MicroConst.a_fr
    b_fr   = MicroConst.b_fr
    γ5br   = MicroConst.γ5br
    ρ0 = MicroConst.ρ0
    α = MicroConst.α
    qc0 = MicroConst.qc0
    Lc = MicroConst.Lc
    Ka = MicroConst.Ka
    Da = MicroConst.Da
    Rvap = PhysConst.Rvap
    Rair = PhysConst.Rair
    μ = MicroConst.μ

    for ip=1:npoin
        # Get state variables depending on lpert/NSD
        if (lpert)
            if (NSD == NSD_3D())
                ρ = uaux[ip,1] + qe[ip,1]
                qt = (uaux[ip,6] + qe[ip,6])/ρ
                qp = (uaux[ip,7] + qe[ip,7])/ρ
            else
                ρ = uaux[ip,1] + qe[ip,1]
                qt = (uaux[ip,5] + qe[ip,5])/ρ
                qp = (uaux[ip,6] + qe[ip,6])/ρ
            end
        else
            if (NSD == NSD_3D())
                ρ = uaux[ip,1]
                qt = uaux[ip,6]/uaux[ip,1]
                qp = uaux[ip,7]/uaux[ip,1]
            else
                ρ = uaux[ip,1]
                qt = uaux[ip,5]/uaux[ip,1]
                qp = uaux[ip,6]/uaux[ip,1]
            end
        end

        T = Tabs[ip]
        e_satw = esatw(T) * 100

        # Only rain microphysics
        # Collection
        Ar_c = π/4 * a_rain * N0_rain * Er_c * γ3br * (ρ0/ρ)^0.5 * (ρ/(π*ρ_rain*N0_rain))^((3+b_rain)/4)
        dqrdt = Ar_c * qc[ip] * qr[ip]^((3+b_rain)/4)

        # Autoconversion (rain from cloud)
        Auto = max(0.0, α * (qc[ip] - qc0))

        # Aggregation: NOT USED in Kessler, set to zero
        Aggr = 0.0

        # Evaporation
        S = max(qt - qn[ip], 0.0) / qsatt[ip]
        A_rain = (Lc / (Ka * T)) * ((Lc / (Rvap * T)) - 1)
        B_rain = Rvap * Rair / (Da * e_satw)
        A_er = a_fr * (ρ / (π * ρ_rain * N0_rain))^0.5
        B_er = b_fr * (ρ * a_rain / μ)^0.5 * γ5br * (ρ0/ρ)^0.25 * (ρ/(π * ρ_rain * N0_rain))^((5+b_rain)/8)
        Evap_r = min(0.0, (2*π*C_rain*N0_rain)/(ρ*(A_rain+B_rain)) * (A_er*sqrt(qr[ip]) + B_er*qr[ip]^((5+b_rain)/8)) * (S-1))

        # Everything else is zero (Kessler does NOT include ice/snow/graupel)
        dqsdt = 0.0
        dqgdt = 0.0
        Evap_s = 0.0
        Evap_g = 0.0

        # Final Kessler microphysical tendency (just rain terms)
        S_micro[ip] = clamp(dqrdt + Auto + Evap_r, -qp, qn[ip])
    end
end


function compute_dqpdt_sam_micro_gpu(u,qe,T,qn,qc,qi,qr,qs,qg,qsatt,MicroConst,PhysConst,lpert)
    FT = eltype(u)
    if (lpert)
        ρ = u[1] + qe[1]
        qt = (u[6] + qe[6]) / ρ
        qp = (u[7] + qe[7]) / ρ
    else
        ρ = u[1]
        qt = u[6] / ρ
        qp = u[7] / ρ
    end

    # Only use rain microphysics constants
    a_rain = MicroConst.a_rain
    b_rain = MicroConst.b_rain
    N0_rain = MicroConst.N0_rain
    Er_c = MicroConst.Er_c
    γ3br = MicroConst.γ3br
    ρ_rain = MicroConst.ρ_rain
    C_rain = MicroConst.C_rain
    a_fr   = MicroConst.a_fr
    b_fr   = MicroConst.b_fr
    γ5br   = MicroConst.γ5br
    ρ0 = MicroConst.ρ0
    α = MicroConst.α
    qc0 = MicroConst.qc0
    Lc = MicroConst.Lc
    Ka = MicroConst.Ka
    Da = MicroConst.Da
    Rvap = PhysConst.Rvap
    Rair = PhysConst.Rair
    μ = MicroConst.μ

    e_satw = esatw(T) * FT(100)

    # Only warm-rain (Kessler) microphysics
    Ar_c = FT(π/4) * a_rain * N0_rain * Er_c * γ3br * (ρ0 / ρ)^FT(0.5) * (ρ / (FT(π) * ρ_rain * N0_rain))^((FT(3) + b_rain) / FT(4))
    dqrdt = Ar_c * qc * qr^( (FT(3) + b_rain) / FT(4) )

    # Autoconversion (rain from cloud)
    Auto = FT(max(FT(0.0), α * (qc - qc0)))

    # No ice/aggregation for Kessler
    Aggr = FT(0.0)

    # Evaporation (rain only)
    S = FT((qt - qn) / qsatt)
    A_rain = (Lc / (Ka * T)) * ((Lc / (Rvap * T)) - FT(1))
    B_rain = Rvap * Rair / (Da * e_satw)
    A_er = a_fr * (ρ / (FT(π) * ρ_rain * N0_rain))^FT(0.5)
    B_er = b_fr * (ρ * a_rain / μ)^FT(0.5) * γ5br * (ρ0 / ρ)^FT(0.25) * (ρ / (FT(π) * ρ_rain * N0_rain))^( (FT(5) + b_rain) / FT(8) )
    Evap_r = FT((FT(2) * FT(π) * C_rain * N0_rain) / (ρ * (A_rain + B_rain))) *
             (A_er * sqrt(qr) + B_er * qr^( (FT(5) + b_rain) / FT(8) )) * (S - FT(1))

    # Snow/graupel/ice terms set to zero
    dqsdt = FT(0.0)
    dqgdt = FT(0.0)
    Evap_s = FT(0.0)
    Evap_g = FT(0.0)

    # Final warm-rain (Kessler) tendency
    dqpdt = FT(clamp(dqrdt + Auto + Evap_r, -qp, qn))

    return FT(dqpdt)
end


function compute_Pm!(uaux, qe, qr, qs, qg, Pr, Ps, Pg, npoin, MicroConst, lpert)
    a_rain = MicroConst.a_rain #Constant in fall speed for rain
    γ4br = MicroConst.γ4br
    ρ_rain = MicroConst.ρ_rain
    N0_rain = MicroConst.N0_rain
    b_rain = MicroConst.b_rain
    ρ0 = MicroConst.ρ0

    a_snow = MicroConst.a_snow #Constant in fall speed for rain
    γ4bs = MicroConst.γ4bs
    ρ_snow = MicroConst.ρ_snow
    N0_snow = MicroConst.N0_snow
    b_snow = MicroConst.b_snow

    a_graupel = MicroConst.a_graupel #Constant in fall speed for rain
    γ4bg = MicroConst.γ4bg
    ρ_graupel = MicroConst.ρ_graupel
    N0_graupel = MicroConst.N0_graupel
    b_graupel = MicroConst.b_graupel
    for ip=1:npoin
        if (lpert)
            ρ = uaux[ip,1] + qe[ip,1] 
        else
            ρ = uaux[ip,1]
        end
        #@info ρ,qr[ip],ρ0
        Pr[ip] = (a_rain * γ4br)/6 * (π * ρ_rain * N0_rain)^(-b_rain/4)*(ρ0/ρ)^(0.5)*(ρ*qr[ip])^(1+b_rain/4)
        Ps[ip] = (a_snow * γ4bs)/6 * (π * ρ_snow * N0_snow)^(-b_snow/4)*(ρ0/ρ)^(0.5)*(ρ*qs[ip])^(1+b_snow/4)
        Pg[ip] = (a_graupel * γ4bg)/6 * (π * ρ_graupel * N0_graupel)^(-b_graupel/4)*(ρ0/ρ)^(0.5)*(ρ*qg[ip])^(1+b_graupel/4)
    end
end


function compute_Pm_gpu(u, qe, qr, qs, qg, MicroConst, lpert)
   
    FT = eltype(u)
    if (lpert)
        ρ = u[1] + qe[1]
    else
        ρ = u[1]
    end
    a_rain = MicroConst.a_rain #Constant in fall speed for rain
    γ4br = MicroConst.γ4br
    ρ_rain = MicroConst.ρ_rain
    N0_rain = MicroConst.N0_rain
    b_rain = MicroConst.b_rain
    ρ0 = MicroConst.ρ0

    a_snow = MicroConst.a_snow #Constant in fall speed for rain
    γ4bs = MicroConst.γ4bs
    ρ_snow = MicroConst.ρ_snow
    N0_snow = MicroConst.N0_snow
    b_snow = MicroConst.b_snow

    a_graupel = MicroConst.a_graupel #Constant in fall speed for rain
    γ4bg = MicroConst.γ4bg
    ρ_graupel = MicroConst.ρ_graupel
    N0_graupel = MicroConst.N0_graupel
    b_graupel = MicroConst.b_graupel

    Pr = (a_rain * γ4br)/FT(6) * (FT(π) * ρ_rain * N0_rain)^(FT(-b_rain/FT(4)))*(ρ0/ρ)^(FT(0.5))*(ρ*qr)^(FT(1)+FT(b_rain/FT(4)))
    Ps = (a_snow * γ4bs)/FT(6) * (FT(π) * ρ_snow * N0_snow)^(FT(-b_snow/FT(4)))*(ρ0/ρ)^(FT(0.5))*(ρ*qs)^(FT(1)+FT(b_snow/FT(4)))
    Pg = (a_graupel * γ4bg)/FT(6) * (FT(π) * ρ_graupel * N0_graupel)^(FT(-b_graupel/FT(4)))*(ρ0/ρ)^(FT(0.5))*(ρ*qg)^(FT(1)+FT(b_graupel/FT(4)))
    #if (Pr > 0)
    #    @info Pr, Ps, Pg
    #end
    return FT(Pr), FT(Ps), FT(Pg)
end

function saturation_adjustment_sam_microphysics!(uaux, qe, Tabs, qn, qi, qc, qr, qs, qg, qsatt, z, npoin, MicroConst, PhysConst, NSD, lpert)

    T00n = MicroConst.T00n #Temperature threshold for cloud water
    T0n  = MicroConst.T0n #Temperature threshold for ice
    T00p = MicroConst.T00p #Temperature threshold for rain     
    T0p  = MicroConst.T0p #Temperature threshold for snow/graupel 
    T00g = MicroConst.T00g #Temperature threshold for graupel
    T0g  = MicroConst.T0g #Temperature threshold for graupel
    
    Lc = MicroConst.Lc
    Lf = MicroConst.Lf
    Ls = MicroConst.Ls
    
    g  = PhysConst.g
    cp = PhysConst.cp
    
    fac_cond = Lc/cp
    fac_fus  = Lf/cp
    fac_sub  = Ls/cp

    for ip=1:npoin
        if (lpert)
            if (NSD == NSD_3D())
                ρ  = uaux[ip,1] + qe[ip,1]
                hl = (uaux[ip,5] + qe[ip,5])/ρ
                qt = (uaux[ip,6] + qe[ip,6])/ρ
                qp = (uaux[ip,7] + qe[ip,7])/ρ
                qt = max(0.0, qt)
                qp = max(0.0, qp)
            
                uaux[ip,6] = qt*ρ - qe[ip,6]
                uaux[ip,7] = qp*ρ - qe[ip,7]
            else

                ρ  = uaux[ip,1] + qe[ip,1]
                hl = (uaux[ip,4] + qe[ip,4])/ρ
                qt = (uaux[ip,5] + qe[ip,5])/ρ
                qp = (uaux[ip,6] + qe[ip,6])/ρ
                qt = max(0.0, qt)
                qp = max(0.0, qp)

                uaux[ip,5] = qt*ρ - qe[ip,5]
                uaux[ip,6] = qp*ρ - qe[ip,6]

            end
        
        else
            if (NSD == NSD_3D())
                ρ  = uaux[ip,1]
                hl = uaux[ip,5]/ρ
                qt = max(0.0,uaux[ip,6])/ρ
                qp = max(0.0,uaux[ip,7])/ρ
            
                uaux[ip,6] = max(0.0,uaux[ip,6])
                uaux[ip,7] = max(0.0,uaux[ip,7])
            else
                ρ  = uaux[ip,1]
                hl = uaux[ip,4]/ρ
                qt = max(0.0,uaux[ip,5])/ρ
                qp = max(0.0,uaux[ip,6])/ρ

                uaux[ip,5] = max(0.0,uaux[ip,5])
                uaux[ip,6] = max(0.0,uaux[ip,6])
            end
        end

        # find equilibrium temperature from saturation adjustment
        # initial guess for sensible temperature and pressure assumes no condensates/all vapor
                
        T =  (hl - g*z[ip])/cp
        an   = 1/(T0n - T00n)
        bn   = T00n * an
        ap   = 1/(T0p - T00p)
        bp   = T00p*ap
        fac1 = fac_cond+(1+bp)*fac_fus
        fac2 = fac_fus*ap
        ag   = 1/(T0g - T00g)
        ωp   = max(0,min(1,ap*T-bp))
        T1   = T + (fac_cond + (1-ωp)*fac_fus)*qp #+ fac1*qp/(1+fac2*qp)
        Tv   = T1*(1 + 0.61*qt - qp)
        P    = moistPressure(PhysConst; ρ=ρ, Tv=Tv, qv = qt)
        if (T1 >= T0p)
            
            T1 = T + fac_cond*qp
            Tv = T1*(1 + 0.61*qt - qp)
            P  = moistPressure(PhysConst; ρ=ρ, Tv=Tv, qv = qt)
        
        elseif (T1 <= T00p)
            
            T1 = T + fac_sub*qp
            Tv = T1*(1 + 0.61*qt - qp)
            P  = moistPressure(PhysConst; ρ=ρ, Tv=Tv, qv = qt)
        
        else
            
            ωp = max(0,min(1,ap*T1-bp))
            T1 = T + (fac_cond + (1-ωp)*fac_fus)*qp
            Tv = T1*(1 + 0.61*qt - qp)
            P  = moistPressure(PhysConst; ρ=ρ, Tv=Tv, qv = qt)
        
        end
        if (T1 >= T0n)

            qsatt[ip] = max(0.0,qsatw(T1, P/100))
        
        elseif (T1 <= T00n)

            qsatt[ip] = max(0.0,qsati(T1, P/100))
        
        else
        
            ωn        = max(0,min(1,an*T1-bn))
            qsatt[ip] = max(0.0,ωn*qsatw(T1,P/100)+(1-ωn)*qsati(T1,P/100))
        
        end
        
        Tabs[ip] = T1

        if (qt > qsatt[ip])
            Tv = T1*(1 + 0.61*min(qt,qsatt[ip]) - qp - max(0,qt-qsatt[ip]))
            P  = moistPressure(PhysConst; ρ=ρ, Tv=Tv, qv = qt)
            
            niter = 0
            dT    = 100
            dqsat = 0.0
            
            while (abs(dT) > 0.0001 && niter < 50)

                if (T1 >= T0n)

                    ωn        = 1
                    lstarn    = fac_cond
                    dlstarn   = 0
                    qsatt[ip] = max(qsatw(T1,P/100),0.0)
                    dqsat     = dtqsatw(T1,P/100)
                    dωn       = 0.0
                
                elseif (T1 <= T00n)

                    ωn        = 0
                    lstarn    = fac_sub
                    dlstarn   = 0
                    qsatt[ip] = max(qsati(T1,P/100),0.0)
                    dqsat     = dtqsati(T1,P/100)
                    dωn       = 0.0
                
                else

                    ωn        = max(0,min(1,an*T1-bn))
                    dωn       = an
                    lstarn    = fac_cond+(1-ωn)*fac_fus
                    dlstarn   = -dωn*fac_fus#dωn*fac_cond - dωn * fac_fus 
                    qsatt[ip] = max(0.0,ωn*qsatw(T1,P/100) + (1-ωn)*qsati(T1,P/100))
                    dqsat     = ωn*dtqsati(T1,P/100) + (1-ωn)*dtqsati(T1,P/100) + dωn * qsatw(T1,P/100) - dωn * qsati(T1,P/100)
                
                end

                if (T1 >= T0p)

                    ωp      = 1
                    lstarp  = fac_cond
                    dlstarp = 0

                elseif (T1 <= T00p)

                    ωp      = 0
                    lstarp  = fac_sub
                    dlstarp = 0

                else

                    ωp      = max(0,min(1,ap*T1-bp))
                    lstarp  = fac_cond + (1-ωp)*fac_fus
                    dlstarp = -ap*fac_fus#ap*fac_cond - ap*fac_fus#ap*fac_fus

                end

                fff   = T - T1 + lstarn*(qt - qsatt[ip]) + lstarp*qp
                dfff  = dlstarn*(qt - qsatt[ip]) - lstarn*dqsat - 1 + dlstarp*qp
                dT    = -fff/dfff
                niter = niter + 1
                T1    = T1 + dT
                Tv    = T1*(1 + 0.61*min(qt,qsatt[ip]) - qp - max(0,qt-qsatt[ip]))
                P     = moistPressure(PhysConst; ρ=ρ, Tv=Tv, qv = qt)
            
            end
            
            qsatt[ip] = qsatt[ip] + dT*dqsat
            qn[ip]    = max(0.0, qt-qsatt[ip])

        else
        
            qn[ip] = 0.0
        
        end

        Tabs[ip] = T1

        ωn = max(0,min(1,an*Tabs[ip]-bn))
        ωp = max(0,min(1,ap*Tabs[ip]-bp))
        ωg = max(0,min(1,ag*Tabs[ip]-T00g*ag))

         # Kessler: Only cloud and rain
        qc[ip] = max(0.0, qn[ip])
        qi[ip] = 0.0
        qr[ip] = max(0.0, qp)
        qs[ip] = 0.0
        qg[ip] = 0.0

        
        uaux[ip,end] = moistPressure(PhysConst; ρ = ρ, Tv = Tv, qv = qt-qn[ip])
    end
end


function saturation_adjustment_sam_microphysics_gpu(u,qe,z,MicroConst,PhysConst,lpert)

    FT = eltype(u)
    if (lpert)
        
        ρ = u[1] + qe[1]
        hl = (u[5] + qe[5])/ρ
        qt = (u[6] + qe[6])/ρ
        qp = (u[7] + qe[7])/ρ
    
    else
    
        ρ = u[1]
        hl = u[5]/ρ
        qt = u[6]/ρ
        qp = u[7]/ρ
    
    end

    T00n = MicroConst.T00n #Temperature threshold for cloud water
    T0n  = MicroConst.T0n #Temperature threshold for ice
    T00p = MicroConst.T00p #Temperature threshold for rain     
    T0p  = MicroConst.T0p #Temperature threshold for snow/graupel 
    T00g = MicroConst.T00g #Temperature threshold for graupel
    T0g  = MicroConst.T0g #Temperature threshold for graupel

    Lc = MicroConst.Lc
    Lf = MicroConst.Lf
    Ls = MicroConst.Ls
    
    g  =PhysConst.g
    cp = PhysConst.cp
    
    fac_cond = Lc/cp
    fac_fus  = Lf/cp
    fac_sub  = Ls/cp
    # find equilibrium temperature from saturation adjustment
    # initial guess for sensible temperature and pressure assumes no condensates/all vapor
    
    T =  (hl - g*z)/cp
    qt = FT(max(FT(0.0),qt))
    qp = FT(max(FT(0.0),qp))
    #P = moistPressure(PhysConst; ρ=ρ, Temp=T, qv = qt) 
    
    an   = FT(1/(T0n - T00n))
    bn   = T00n * an
    ap   = FT(1/(T0p - T00p))
    bp   = T00p*ap
    fac1 = fac_cond+(FT(1)+bp)*fac_fus
    fac2 = fac_fus*ap
    ag   = FT(FT(1)/(T0g - T00g))
    T1   = T + fac1*qp/(FT(1)+fac2*qp)
   
    Tv   = T1*(1 + 0.61*qt - qp)
    P    = moistPressure(PhysConst; ρ=ρ, Tv=Tv, qv = qt)

    if (T1 >= T0p)

        T1 = T + fac_cond*qp
        Tv = T1*(1 + 0.61*qt - qp)
        P  = moistPressure(PhysConst; ρ=ρ, Tv=Tv, qv = qt)

    elseif (T1 <= T00p)

        T1 = T + fac_sub*qp
        Tv = T1*(1 + 0.61*qt - qp)
        P  = moistPressure(PhysConst; ρ=ρ, Tv=Tv, qv = qt)

    else

        ωp = max(0,min(1,ap*T1-bp))
        T1 = T + (fac_cond + (1-ωp)*fac_fus)*qp
        Tv = T1*(1 + 0.61*qt - qp)
        P  = moistPressure(PhysConst; ρ=ρ, Tv=Tv, qv = qt)

    end

    if (T1 >= T0n)

        qsatt = FT(max(0.0,qsatw(T1, P/100)))

    elseif (T1 <= T00n)

        qsatt = FT(max(0.0,qsati(T1, P/100)))

    else

        ωn    = FT(max(0,min(1,an*T1-bn)))
        qsatt = FT(max(0.0,ωn*qsatw(T1,P/100)+(1-ωn)*qsati(T1,P/100)))

    end

    if (qt > qsatt)

        niter = Int32(0)
        dT    = FT(100)
        dqsat = FT(0.0)
        
        while (abs(dT) > FT(0.001) && niter < FT(50))

            if (T1 >= T0n)

                ωn      = 1
                lstarn  = fac_cond
                dlstarn = FT(0)
                qsatt   = FT(qsatw(T1,P/100))
                dqsat   = FT(dtqsatw(T1,P/100))

            elseif (T1 <= T00n)

                ωn      = FT(0)
                lstarn  = fac_sub
                dlstarn = FT(0)
                qsatt   = FT(qsati(T1,P/100))
                dqsat   = FT(dtqsati(T1,P/100))

            else

                ωn      = FT(max(FT(0),FT(min(FT(1),an*T1-bn))))
                lstarn  = fac_cond+(FT(1)-ωn)*fac_fus
                dlstarn = an*fac_fus
                qsatt   = FT(max(FT(0.0),ωn*FT(qsatw(T1,P/100)) + (FT(1)-ωn)*FT(qsati(T1,P/100))))
                dqsat   = ωn*FT(dtqsati(T1,P/100)) + (FT(1)-ωn)*FT(dtqsati(T1,P/100)) + dωn * qsatw(T1,P/100) - dωn * qsati(T1,P/100)
            end

            if (T1 >= T0p)

                ωp      = FT(1)
                lstarp  = fac_cond
                dlstarp = FT(0)

            elseif (T1 <= T00p)

                ωp      = FT(0)
                lstarp  = fac_sub
                dlstarp = FT(0)

            else

                ωp      = FT(max(FT(0),FT(min(FT(1),ap*T1-bp))))
                lstarp  = fac_cond + (FT(1)-ωp)*fac_fus
                dlstarp = ap*fac_fus

            end

            fff   = T - T1 + lstarn*(qt - qsatt) + lstarp*qp
            dfff  = dlstarn*(qt - qsatt) - lstarn*dqsat - FT(1) + dlstarp*qp
            dT    = -fff/dfff
            niter = niter + Int32(1)
            T1    = T1 + dT
            Tv    = T1*(FT(1) + FT(0.61)*min(qt,qsatt) - qp - max(FT(0),qt-qsatt))
            P     = moistPressure(PhysConst; ρ=ρ, Tv=Tv, qv = qt)
        end

        qsatt = qsatt + dqsat * dT
        qn    = max(FT(0.0), qt-qsatt)

    else

        qn = FT(0.0)
    end

    T = T1

    qp = FT(max(FT(0.0), qp))
    ωn = FT(max(FT(0),FT(min(FT(1),an*T-bn))))
    ωp = FT(max(FT(0),FT(min(FT(1),ap*T-bp))))
    ωg = FT(max(FT(0),FT(min(FT(1),ag*T-T00g*ag))))

    qc = FT(max(FT(0.0), qn))
    qi = FT(0.0)
    qr = FT(max(FT(0.0), qp))
    qs = FT(0.0)
    qg = FT(0.0)
    P = moistPressure(PhysConst; ρ = ρ, Temp = T, qv = qt-qn)
    return FT(T),FT(P),FT(qn),FT(qc),FT(qi),FT(qr),FT(qs),FT(qg),FT(qsatt)

end