using Arrow
using DataFrames
using CSV
using Dierckx
using Plots
using JLD
using SparseArrays
using FLOWMath
using PolyFit
using Statistics
using Base.Threads
using Distributions
using Interpolations
using CubicSplines
using FLOWMath
using ImageFiltering
using ForwardDiff
using SNOW
using LaTeXStrings
# using StatsPlots
"""
This script takes the filename specificied by the user to look at specified XZ and YZ slices in SOWFA Data
The slices of data are prepped for plotting in contour plots as well as saving to be compared to with 
FLOWFarm results and FLORIS results.

XZ slice and YZ slice data are saved as .jld files named by filename and filenameYZ respectively.
These files can be opened up to be compared with FLORIS and FLOWFARM simulations.

"""

mutable struct Param_val{}
    x_data
    y_data
end

function EMG(x, mean, sigma_square, lambda)
    lambda = 1/lambda
    b = (lambda/2) * (2*mean + lambda*sigma_square .- 2*x)
    c = (mean + lambda*sigma_square .- x)/(sqrt(2)*sqrt(sigma_square))
    fx = (lambda/2)*(exp.(b)).*(erfc.(c))
    return fx
end

function EMG_optimization(x)
    x = abs.(x)
    lambda = 1/x[4]
    mean = x[2]
    sigma_square = x[3]
    C = x[1]
    X_val = [5.4878048780487800
    8.536585365853650
    11.585365853658500
    16.463414634146300
    20.731707317073200
    23.78048780487810
    27.4390243902439
    31.097560975609800
    35.36585365853660
    42.07317073170730
    46.34146341463420
    52.439024390243900
    59.14634146341460
    68.29268292682930
    78.04878048780490
    85.36585365853660
    100.60975609756100
    110.97560975609800
    119.51219512195100
    126.82926829268300
    135.97560975609800
    146.9512195121950
    159.7560975609760
    167.07317073170700
    176.82926829268300
    184.14634146341500
    192.6829268292680
    200.60975609756100
    209.7560975609760
    213.41463414634100
    221.95121951219500
    229.87804878048800
    240.85365853658500
    250.00000000000000
    262.19512195122000
    271.3414634146340
    278.6585365853660
    292.68292682926800
    304.8780487804880]
    Y_val = [0.032786885245901500
    0.11475409836065600
    0.21311475409836100
    0.3770491803278690
    0.516393442622951
    0.6516393442622950
    0.7786885245901640
    0.8565573770491810
    0.9836065573770490
    1.0901639344262300
    1.1598360655737700
    1.2213114754098400
    1.270491803278690
    1.3155737704918000
    1.3565573770491800
    1.3729508196721300
    1.364754098360660
    1.34016393442623
    1.3155737704918000
    1.2786885245901600
    1.2131147540983600
    1.1229508196721300
    1.0409836065573800
    0.9959016393442630
    0.9180327868852460
    0.8524590163934430
    0.7704918032786890
    0.7131147540983610
    0.627049180327869
    0.5942622950819670
    0.5122950819672130
    0.4508196721311480
    0.3606557377049180
    0.29508196721311500
    0.20491803278688500
    0.15573770491803300
    0.10245901639344300
    0.06557377049180310
    0.040983606557377000]
    b = (lambda/2) * (2*mean + lambda*sigma_square .- 2*X_val)
    c = (mean + lambda*sigma_square .- X_val)/(sqrt(2)*sqrt(sigma_square))
    fx = C*(lambda/2)*(exp.(b)).*(erfc.(c))
    diff = sum((Y_val.-fx).^2)
    return diff
end

function skewnormal_opt(x)
    mu = x[1]
    sigma = x[2]
    alpha = x[3]
    CC = x[4]
    XX = [5.4878048780487800
    8.536585365853650
    11.585365853658500
    16.463414634146300
    20.731707317073200
    23.78048780487810
    27.4390243902439
    31.097560975609800
    35.36585365853660
    42.07317073170730
    46.34146341463420
    52.439024390243900
    59.14634146341460
    68.29268292682930
    78.04878048780490
    85.36585365853660
    100.60975609756100
    110.97560975609800
    119.51219512195100
    126.82926829268300
    135.97560975609800
    146.9512195121950
    159.7560975609760
    167.07317073170700
    176.82926829268300
    184.14634146341500
    192.6829268292680
    200.60975609756100
    209.7560975609760
    213.41463414634100
    221.95121951219500
    229.87804878048800
    240.85365853658500
    250.00000000000000
    262.19512195122000
    271.3414634146340
    278.6585365853660
    292.68292682926800
    304.8780487804880]
    Y_val = [0.032786885245901500
    0.11475409836065600
    0.21311475409836100
    0.3770491803278690
    0.516393442622951
    0.6516393442622950
    0.7786885245901640
    0.8565573770491810
    0.9836065573770490
    1.0901639344262300
    1.1598360655737700
    1.2213114754098400
    1.270491803278690
    1.3155737704918000
    1.3565573770491800
    1.3729508196721300
    1.364754098360660
    1.34016393442623
    1.3155737704918000
    1.2786885245901600
    1.2131147540983600
    1.1229508196721300
    1.0409836065573800
    0.9959016393442630
    0.9180327868852460
    0.8524590163934430
    0.7704918032786890
    0.7131147540983610
    0.627049180327869
    0.5942622950819670
    0.5122950819672130
    0.4508196721311480
    0.3606557377049180
    0.29508196721311500
    0.20491803278688500
    0.15573770491803300
    0.10245901639344300
    0.06557377049180310
    0.040983606557377000]
    X_val = (XX.-mu)/sigma
    XXX = (X_val.^2)/2
    A = 1/(sigma*sqrt(2*pi))
    B = exp.(-XXX)
    C = 0.5*(1.0 .+erf.(alpha*X_val/sqrt(2)))
    fx = CC*(2*A)*(B.*C)
    diff = sum((Y_val.-fx).^2)
    return diff
end
    

function skewnormal(x, mu, sigma, alpha)
    X_val = (x.-mu)/sigma
    XX = (X_val.^2)/2
    A = 1/(sigma*sqrt(2*pi))
    B = exp.(-XX)
    C = 0.5*(1.0 .+erf.(alpha*X_val/sqrt(2)))
    fx = (2*A)*(B.*C)
    return fx
end

function movingaverage(X::Vector,numofele::Int)
    BackDelta = div(numofele,2) 
    ForwardDelta = isodd(numofele) ? div(numofele,2) : div(numofele,2) - 1
    len = length(X)
    Y = similar(X)
    for n = 1:len
        lo = max(1,n - BackDelta)
        hi = min(len,n + ForwardDelta)
        Y[n] = mean(X[lo:hi])
    end
    return Y
end

function gridpoint(TD, dx)
    Minus_5 = TD - 5.0
    multiple = Minus_5/dx
    point = 5 + (10*round(multiple))
    return point
end

function neg_filter(negative)
    QQQ = size(negative)
    ambience = hcat(negative[:,1:5], negative[:,48:51])
    average = mean(ambience, dims=2)
    QQQ = size(negative)
    negative_negative = average
    for i in 1:QQQ[2]-1
        negative_negative = hcat(negative_negative, average)
    end

    negative = negative_negative - negative
end

function ambient_flow_solve(negative)
    QQQ = size(negative)
    ambience = hcat(negative[:,1:5], negative[:,48:51])
    average = mean(ambience, dims=2)
    QQQ = size(negative)
    negative_negative = average
    for i in 1:QQQ[2]-1
        negative_negative = hcat(negative_negative, average)
    end

    return negative_negative
end

# function required for manual gaussian fitting to SOWFA data
function gaussfitting!(g, Var, parameters)
    # x = 5:10:305        # Should this be finely spaced?
    # params = 5:10:305 
    # print("params_in_function: ", parameters, "\n")
    a = 1/(Var[2]*sqrt(2*pi))
    b = @. exp(-((parameters.x_data-Var[1])^2)/(2*Var[2]^2))
    u = @. a*b
    diff_vals = @. (parameters.y_data - Var[3]*u)^2
    return sum(diff_vals)
end

# for plotting gaussian fit
function gaussvalues(σ, μ, A, x)
    # x = 5:10:305        # Should this be finely spaced?
    # params = 5:10:305 

    a = 1/(σ*sqrt(2*pi))
    b = @. exp(-((x-μ)^2)/(2*σ^2))
    u = @. A*a*b
    return u
end

# wrapper function for objective of gauss fitting
obj_func!(g, Var) = gaussfitting!(g, Var, parameters)

function fit_gauss(parameters)
    
    x0 = [80.0, 50.0, 100.0]
    # IPOPT options
    # print("params: ", params, "\n")
    ip_options = Dict("max_iter" => 1000, "tol" => 1e-6)
    solver = IPOPT(ip_options)
    options = Options(solver=solver, derivatives=ForwardAD())
    # options = Options()
    # print("options: ", options, "\n")
    ng = 2
    lx = -Inf
    ux = Inf
    lg=-Inf
    ug=Inf
    xopt, fopt, info, out = minimize(obj_func!, x0, ng, lx, ux, lg, ug, options)
    print("fopt: ", fopt, "\n")
    print("xopt: ", xopt, "\n")
    # struct gfit_val{}
    #     σ
    #     μ
    # end
    # gfit_final = gfit_val(xopt[2], xopt[1])

    gfit_final = Dict("σ" => xopt[2], "μ" => xopt[1], "A" => xopt[3])
    # for plotting
    X_plot = parameters.x_data
    Gfit_plot = gaussvalues(gfit_final["σ"], gfit_final["μ"], gfit_final["A"], X_plot)
    plot(parameters.x_data, parameters.y_data, label="SOWFA data")
    plot!(X_plot, Gfit_plot, label="Gaussian Fit")
    savefig("gauss_fit_check.png")

    plot(X_plot, Gfit_plot)
    savefig("gaussfit_check2.png")
    return gfit_final
end

function fit_gaussy(parameters)
    
    x0 = [2500.0, 50.0, 100.0]
    # IPOPT options
    # print("params: ", params, "\n")
    ip_options = Dict("max_iter" => 1000, "tol" => 1e-6)
    solver = IPOPT(ip_options)
    options = Options(solver=solver, derivatives=ForwardAD())
    # options = Options()
    # print("options: ", options, "\n")
    ng = 2
    lx = -Inf
    ux = Inf
    lg=-Inf
    ug=Inf
    xopt, fopt, info, out = minimize(obj_func!, x0, ng, lx, ux, lg, ug, options)
    print("fopt: ", fopt, "\n")
    print("xopt: ", xopt, "\n")
    # struct gfit_val{}
    #     σ
    #     μ
    # end
    # gfit_final = gfit_val(xopt[2], xopt[1])

    gfit_final = Dict("σ" => xopt[2], "μ" => xopt[1], "A" => xopt[3])
    # for plotting
    X_plot = parameters.x_data
    Gfit_plot = gaussvalues(gfit_final["σ"], gfit_final["μ"], gfit_final["A"], X_plot)
    plot(parameters.x_data, parameters.y_data, label="SOWFA data")
    plot!(X_plot, Gfit_plot, label="Gaussian Fit")
    savefig("gauss_fit_checky.png")

    plot(X_plot, Gfit_plot)
    savefig("gaussfit_checky2.png")
    return gfit_final
end

function deflection(C_T, d, gamma, ky, kz, sig0y, sig0z, theta_c0, x, I, alpha, Beta)
    # Constant parameters
    # alpha = 2.32;
    # Beta = 0.154;

    x_0 = d*cos(gamma)*(1 + sqrt(1-C_T))/(sqrt(2)*(alpha*I+Beta*(1-sqrt(1-C_T))))
    x = x_0:x;
    sigy = ky*(x.-x_0) .+ d*sig0y
    sigy = abs.(sigy)
    sigz = kz*(x.-x_0) .+ d*sig0z
    sigz = abs.(sigz)
    # print("sigy: ", sigy, "\n")
    # print("sigz: ", sigz, "\n")
    # print("(d^2 * cos(gamma)): ", (d^2 * cos(gamma)), "\n")
    # print("1.6*sqrt.((8 *sigy.*sigz)/(d^2 * cos(gamma))): ", 1.6*sqrt.((8 *sigy.*sigz)/(d^2 * cos(gamma))))
    a = (1.6 + sqrt(C_T))*(1.6*sqrt.((8 *sigy.*sigz)/(d^2 * cos(gamma))) .- sqrt(C_T))
    b = (1.6 - sqrt(C_T))*(1.6*sqrt.((8*sigy.*sigz)/(d^2 * cos(gamma))) .+ sqrt(C_T))
    # print("astuff: ", 1.6*sqrt.((8 *sigy.*sigz)/(d^2 * cos(gamma))) .- sqrt(C_T), "\n")
    c = theta_c0*(x_0)/d
    e = (theta_c0/14.7)*sqrt(cos(gamma)/(ky*kz*C_T))*(2.9-1.3*sqrt(1-C_T)-C_T)
    # print("a./b: ", a./b, "\n")
    defl = d*(c .+ e*log.(a./b))
    return defl, x
end

function deflection_call(index, ky, kz, sig0y, sig0z, TILT, I, C_T, alpha, beta, x_end)
    ky = ky[index]
    kz = kz[index]
    d = 126.0
    # kz2 = -0.003
    # # sigma0y = 0.15441985668899924
    # # sigma0x = 0.29874
    # d = 126.0
    # # Bastankhah numbers
    tilt = TILT[index] * pi/180     # degrees
    x = d*x_end;

    sigma0y = sig0y[index]   # 1/sqrt(8)
    sigma0z = sig0z[index]      # cos(tilt)/sqrt(8)


    theta_c0 = ((0.3*tilt)/cos(tilt))*(1-sqrt(1-C_T*cos(tilt)))

    delf, x_var = deflection(C_T, d, tilt, abs(ky), abs(kz), sigma0y, sigma0z, theta_c0, x, I, alpha, beta)


    return delf, x_var
end

function kykzint(sigmay, sigmaz, X_D_y, X_D_z, beginning)
    # # Y_direction

    Q = polyfit(X_D_y[beginning:end], sigmay[beginning:end], 1)
    ky = Q[1]
    eps = Q[0]
    Z_values = eps.+ky.*X_D_y[beginning:end]

    # # Z_direction
    Qz = polyfit(X_D_z, sigmaz, 1)
    kz = Qz[1]
    epsz = Qz[0]
    Y_values = epsz.+kz.*X_D_z
    ky = round(ky, digits=3)
    kz = round(kz, digits=3)

    return Z_values, Y_values, ky, kz, epsz, eps
end

function SOWFA_analysis(slices, movname, gauss_namez, gauss_namey, tol_comp, file_1, smoothing, tilt)
    sigmaz = zeros(length(slices),length(smoothing))
    sigmaz_ground = zeros(length(slices),length(smoothing))
    sigmay = zeros(length(slices),length(smoothing))
    defz = zeros(length(slices),length(smoothing))
    defy = zeros(length(slices),length(smoothing))
    cup = zeros(length(slices),length(smoothing))
    cdown = zeros(length(slices),length(smoothing))
    power = zeros(length(slices),length(smoothing))
    power_up = zeros(length(slices),length(smoothing))
    avg_V = zeros(length(slices),length(smoothing))
    for j in 1:length(smoothing)
        avg_factor = smoothing[j]
        for i in 1:length(slices)
            """ Filename identifiers for data to be saved to """
            # filename = "base.jld"
            filenameYZ = "YZ_slice_tilt_opp.jld"
            downstream_loc = 9

            """Turbine Locations"""
            # In front of next turbine by .45*D
            turbine1 = 1325.0       # Wake of turbine1 before it reaches rotor of turbine2

            # Turbine Positions in X-direction
            T_pos_1 = 505.0

            # Rotor diameter
            D = 126.0       # meters

            # Getting Multiple dowstream locations behind third turbine
            T_pos = T_pos_1      # turbine position for looking at downstream positions

            inflow = 435.0          # Inflow location to upstream turbine

            # y_center = 905.0        # Center y location to indicate center of turbines
            y_center = 2505.0        # Center y location to indicate center of turbines

            """Which turbine to see YZ slice"""
            # 14
            # 10
            # 6
            print("SLICE: ", slices[i], "\n")
            slice = gridpoint((T_pos + slices[i]*D), 10.0)
            moviename = join([movname, "_", string(slices[i]), "_", smoothing[j], "_n_no_legend", ".pdf"])
            power_filename = join([movname, "_", string(slices[i]), "_", smoothing[j], "_power_slice_downstream", ".pdf"])
            power_filename_down = join([movname, "_", string(slices[i]), "_", smoothing[j], "_power_slice_upstream", ".pdf"])
            gauss_name_y = join([gauss_namey, "_", string(slices[i]), "_", smoothing[j], "_n", ".pdf"])
            gauss_name_z = join([gauss_namez, "_", string(slices[i]), "_", smoothing[j], "_n", ".pdf"])
            gauss_name_z_full = join([gauss_namez, "_", string(slices[i]), "_", smoothing[j], "full", "_n", ".pdf"])
            tolerance_comp = tol_comp       # m/s       # minimum seems to be 0.5

            # ideas for a moving threshold
            # make threshold decrease the same amount as the velocity deficit is decreasing
            # So start with a higher tolerance_comp (like 1.5) then descrease based on how much the wake is recovering


            slice_comp = gridpoint((T_pos - 1*D), 10.0)
            # print("slice_comp: ", slice_comp)
            index_vel = 2
            maxv = 9
            minv = 8
            velocity = minv:(maxv-minv)/8:maxv
            velocity = velocity[index_vel]
            Z_cut_off = 0

            # if you change x_low and x_high you will also need to change 1:5 and 48:51 in the neg_filter
            x_low = 2250
            x_high = 2750

            # Freestream velocity to get (U_inf - U_c) for sigma calculation
            U_inf = 8.05     # m/s       Freestream at hubheight


            # As the wake recovers there is an influence on our measurements
            # this means that tolerance_comp should probably also be adjusted?
            # but how do I do that without influencing the results by the model used to change tolerance_comp


            """Import Data"""
            table1 = Arrow.Table(file_1)
            table1 = DataFrame(table1)

            Num_rows = size(table1[:,1])
            Num_rows = Num_rows[1]

            # Find XZ slice of three turbine array
            XZ_X = []
            XZ_Z = []
            XZ_u = []

            # Find XY slice of most upstream turbine
            YZ_X = []
            YZ_Y = []
            YZ_u = []

            YZ_X_comp = []
            YZ_Y_comp = []
            YZ_u_comp = []

            """ Reorder data into rows and columns for indicated slices"""

            for i in 1:Num_rows
                x_value = table1[i,1]
                y_value = table1[i,2]
                z_value = table1[i,3]
                u_value = table1[i,4]
                v_value = table1[i,5]
                w_value = table1[i,6]

                """ For XZ data slice (streamwise) """
                # turbines are aligned at y = 2500 meters
                if y_value == y_center
                    push!(XZ_X, x_value)
                    push!(XZ_Z, z_value)
                    push!(XZ_u, u_value)
                end

                """ For YZ data slice """
                # print("x_value: ", x_value, "\n")
                # print("inflow: ", inflow, "\n")
                if x_value == slice
                    push!(YZ_X, z_value)
                    push!(YZ_Y, y_value)
                    push!(YZ_u, u_value)
                end

                if x_value == slice_comp
                    push!(YZ_X_comp, z_value)
                    push!(YZ_Y_comp, y_value)
                    push!(YZ_u_comp, u_value)
                end

            end

            XZ_X = parse.(Float64, string.(XZ_X))
            XZ_Z = parse.(Float64, string.(XZ_Z))
            XZ_u = parse.(Float64, string.(XZ_u))

            YZ_X = parse.(Float64, string.(YZ_X))
            YZ_Y = parse.(Float64, string.(YZ_Y))
            YZ_u = parse.(Float64, string.(YZ_u))

            YZ_X_comp = parse.(Float64, string.(YZ_X_comp))
            YZ_Y_comp = parse.(Float64, string.(YZ_Y_comp))
            YZ_u_comp = parse.(Float64, string.(YZ_u_comp))

            """Find max and min of these slices"""
            # XZ slice
            x_min = findmin(XZ_X)
            x_max = findmax(XZ_X)
            z_min = findmin(XZ_Z)
            z_max = findmax(XZ_Z)

            # YZ slice
            x_min_yz = findmin(YZ_X)
            x_max_yz = findmax(YZ_X)
            y_min = findmin(YZ_Y)
            y_max = findmax(YZ_Y)

            """ Determinie Delta Y (Assuming evenly spaced grid) """
            spacing_yz = YZ_Y[2] - YZ_Y[1]

            """ Determinie Delta X (Assuming evenly spaced grid) """
            spacing = XZ_X[2]-XZ_X[1]

            """ Initialize XZ and YZ grids for contour plots """
            X_plot_yz = x_min_yz[1]:spacing_yz:x_max_yz[1]
            Y_plot = y_min[1]:spacing_yz:y_max[1]

            X_plot = x_min[1]:spacing:x_max[1]
            Z_plot = z_min[1]:spacing:z_max[1]

            NumX_yz = (x_max_yz[1]-x_min_yz[1])/spacing_yz
            NumY_yz = (y_max[1] - y_min[1])/spacing_yz

            NumX = (x_max[1]-x_min[1])/spacing
            NumZ = (z_max[1]-z_min[1])/spacing

            NumX = convert(Int64, NumX)
            NumZ = convert(Int64, NumZ)

            NumX_yz = convert(Int64, NumX_yz)
            NumY_yz = convert(Int64, NumY_yz)

            X_plot = LinRange(x_min[1], x_max[1], NumX+1)
            Z_plot = LinRange(z_min[1], z_max[1], NumZ+1)
            X_plot_yz = LinRange(x_min_yz[1], x_max_yz[1], NumX_yz+1)
            Y_plot = LinRange(y_min[1], y_max[1], NumY_yz+1)

            """ Create grid, assuming grid is uniformly spaced """
            size_X = size(X_plot)
            size_X_yz = size(Y_plot)
            size_Z = size(X_plot)
            size_Y = size(Y_plot)

            YZ_grid = zeros((size_X_yz[1], size_Y[1]))
            YZ_grid_comp = zeros((size_X_yz[1], size_Y[1]))
            XZ_grid = zeros((size_X[1], size_Z[1]))

            size_U_yz = size(YZ_u)
            size_U = size(XZ_u)
            spacing_yz = convert(Int64, spacing_yz)
            spacing = convert(Int64, spacing)

            """Fill XZ grid values"""

            for i in 1:size_U[1]
                x_coord = XZ_X[i]
                z_coord = XZ_Z[i]
                x_coord = convert(Int64, x_coord)
                z_coord = convert(Int64, z_coord)
                x_coord = ((x_coord-(spacing/2))/spacing)+1
                z_coord = ((z_coord-(spacing/2))/spacing)+1
                x_coord = convert(Int64, x_coord)
                z_coord = convert(Int64, z_coord)
                XZ_grid[x_coord, z_coord] = XZ_u[i]
            end

            """Fill YZ grid values"""

            for i in 1:size_U_yz[1]
                x_coord_yz = YZ_X[i]
                y_coord_yz = YZ_Y[i]
                x_coord_yz = convert(Int64, x_coord_yz)
                y_coord_yz = convert(Int64, y_coord_yz)
                x_coord_yz = ((x_coord_yz-(spacing_yz/2))/spacing_yz)+1
                y_coord_yz = ((y_coord_yz-(spacing_yz/2))/spacing_yz)+1
                x_coord_yz = convert(Int64, x_coord_yz)
                y_coord_yz = convert(Int64, y_coord_yz)
                YZ_grid[x_coord_yz, y_coord_yz] = YZ_u[i]
            end

            """Fill YZ_comp grid values"""

            for i in 1:size_U_yz[1]
                x_coord_yz = YZ_X_comp[i]
                y_coord_yz = YZ_Y_comp[i]
                x_coord_yz = convert(Int64, x_coord_yz)
                y_coord_yz = convert(Int64, y_coord_yz)
                x_coord_yz = ((x_coord_yz-(spacing_yz/2))/spacing_yz)+1
                y_coord_yz = ((y_coord_yz-(spacing_yz/2))/spacing_yz)+1
                x_coord_yz = convert(Int64, x_coord_yz)
                y_coord_yz = convert(Int64, y_coord_yz)
                YZ_grid_comp[x_coord_yz, y_coord_yz] = YZ_u_comp[i]
            end


            # """ Plot XZ """
            XZ_grid = transpose(XZ_grid)

            # Save YZ slice
            save(filenameYZ, "data", YZ_grid)
            # data1 = heatmap(Y_plot, Y_plot, YZ_grid)

            # Comparing YZ slices
            ### Load SOWFA data
            SOWFA_data = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/"*filenameYZ
            SOWFA_data = load(SOWFA_data)["data"]

            ### Load FLOWFarm data
            # FLOW_data = "/Users/jamescutler/Downloads/bbyu_tilt_runs_single/FLOWFarm/YZ_slice_turbine_3_base.jld"
            # FLOW_data = load(FLOW_data)["data"]

            # ## Load FLORIS data
            # FLORIS_data = "/Users/jamescutler/Downloads/byu_tilt_runs_single/FLORIS_runs/YZslice_25_25_floris_umesh.dat"
            # FLORIS_data = CSV.read(FLORIS_data, DataFrame)
            # FLORIS_data = Matrix{Union{Real,Missing}}(FLORIS_data)
            # difference = SOWFA_data.-FLOW_data
            # # difference = SOWFA_data

            # data1 = heatmap(Y_plot, Y_plot, clim=(-1,1), c = :bluesreds, difference)
            # plot(data1, ylim=(0,300), xlim=(700,1150), aspect_ratio=:equal)

            # # plot rotor radius
            f(t) = 900 .+ (126/2).*cos.(t)
            g(t) = 90 .+ (126/2).*sin.(t)
            range = 0:0.01:2*pi
            x_values = f(range)
            y_value = g(range)
            plot!(x_values, y_value)

            # print(size(Y_plot), "\n")
            # print(size(SOWFA_data))

            # data1 = heatmap(Y_plot, Y_plot, clim=(0,9), SOWFA_data)
            # plot(data1, ylim=(0,300), xlim=(700,1150), aspect_ratio=:equal)

            """crop SOWFA data to find center"""
            Z_limits = [300, Z_cut_off]
            X_limits = [x_low, x_high]

            znum = round((Z_limits[1] - 5)/10)
            zn1 = Int(1+znum)
            znum = round((Z_limits[2] - 5)/10)
            zn2 = Int(1+znum)

            xnum = round((X_limits[1] - 5)/10)
            xn1 = Int(1+xnum)
            xnum = round((X_limits[2] - 5)/10)
            xn2 = Int(1+xnum)

            cropped_SOWFA_data = SOWFA_data[zn2:zn1, xn1:xn2]
            cropped_YZ_grid_comp = YZ_grid_comp[zn2:zn1, (xn1-100):(xn2-100)]
            crop_Y_plot = Y_plot[xn1:xn2]
            crop_Z_plot = Y_plot[zn2:zn1]
            print("crop_Y_plot: ", crop_Y_plot, "\n")
            print("crop_Z_plot: ", crop_Z_plot, "\n")


            # # cropped_YZ_grid_comp needs to be averaged across more areas
            # negative = cropped_YZ_grid_comp - cropped_SOWFA_data

            # filter out spurious velocities near the ground of the wake
            negative = neg_filter(cropped_SOWFA_data)
            ambient_flow = ambient_flow_solve(cropped_SOWFA_data)

            # cropped_SOWFA_data_comp = SOWFA_data[zn2:zn1, xn1:xn2]
            cropped_SOWFA_data_comp_neg = negative[:, :]
            YZ_grid_comp = YZ_grid_comp[zn2:zn1, xn1-100:xn2-100]
            crop_Y_plot_comp = Y_plot[xn1:xn2]
            crop_Z_plot_comp = Y_plot[zn2:zn1] 

            # finenum = 15000
            finenum = 1000

            
            # Interpolation
            # twodinterpol = LinearInterpolation((crop_Z_plot_comp, crop_Y_plot_comp),cropped_SOWFA_data_comp_neg)
            # print("crop_Y_plot_comp: ", crop_Y_plot_comp)
            # crop_Y_plot_fine = range(extrema(crop_Y_plot_comp)..., length=finenum)
            crop_Y_plot_fine = minimum(crop_Y_plot_comp):((maximum(crop_Y_plot_comp)-minimum(crop_Y_plot_comp))/finenum):maximum(crop_Y_plot_comp)
            # print("crop_Y_plot_fine: ", crop_Y_plot_fine)
            # crop_Z_plot_fine = range(extrema(crop_Z_plot_comp)..., length=finenum)
            crop_Z_plot_fine = minimum(crop_Z_plot_comp):((maximum(crop_Z_plot_comp)-minimum(crop_Z_plot_comp))/finenum):maximum(crop_Z_plot_comp)
            # cropped_SOWFA_data_comp_neg_fine = [twodinterpol(x,y) for x in crop_Z_plot_fine, y in crop_Y_plot_fine]

            # Interpolate data using FLOWMath
            crop_Y_plot_fine = convert(Vector{Float64}, crop_Y_plot_fine)
            crop_Z_plot_fine = convert(Vector{Float64}, crop_Z_plot_fine)
            crop_Y_plot_comp = convert(Vector{Float64}, crop_Y_plot_comp)
            crop_Z_plot_comp = convert(Vector{Float64}, crop_Z_plot_comp)
            print(size(crop_Z_plot_fine))
            print(typeof(cropped_SOWFA_data_comp_neg))
            


            cropped_SOWFA_data_comp_neg_fine = interp2d(akima, crop_Z_plot_comp, crop_Y_plot_comp, cropped_SOWFA_data_comp_neg, crop_Z_plot_fine, crop_Y_plot_fine)
            cropped_SOWFA_data_comp_fine = interp2d(akima, crop_Z_plot_comp, crop_Y_plot_comp, cropped_SOWFA_data, crop_Z_plot_fine, crop_Y_plot_fine)
            cropped_SOWFA_data_ambient_fine = interp2d(akima, crop_Z_plot_comp, crop_Y_plot_comp, ambient_flow, crop_Z_plot_fine, crop_Y_plot_fine)

            # Smooth out 2-d data with gaussian filter to remove block remnants
            # should this happen before or after interpolation?
            cropped_SOWFA_data_comp_neg_fine = imfilter(cropped_SOWFA_data_comp_neg_fine, Kernel.gaussian(40))


            finestep = ((maximum(crop_Z_plot_comp)-minimum(crop_Z_plot_comp))/finenum)
            """Test to find all peaks in slice"""
            """If there are multiple peaks then use Threshold hybrid method"""
            """Otherwise it means we are significantly downstream that we can just find maximum velocity-deficit"""
            """Or if we are significantly downstream (dist>7D) and there are multiple peaks it means there is a kidney bean shape"""

            """Find maximum points in wake"""
            Global_max = findmax(cropped_SOWFA_data_comp_neg_fine)
            Y_global_max = crop_Y_plot_fine[Global_max[2][2]]
            Z_global_max = crop_Z_plot_fine[Global_max[2][1]]
            print("center_Y: ", Y_global_max, "\n")
            print("center_Z: ", Z_global_max)
            Global_Horz_profile = cropped_SOWFA_data_comp_neg_fine[Global_max[2][1],:]
            Global_Vert_profile = cropped_SOWFA_data_comp_neg_fine[:,Global_max[2][2]]
            

            """Find maximum points in original data"""


            """Trim Profile for gaussian fit"""
            # anything in profiles above 0.1 helps get rid of edges
            Trim_edges_horz = findall(t -> t<0.8, Global_Horz_profile)
            Trim_edges_vert = findall(t -> t<0.8, Global_Vert_profile)
        
            # Remove edges of velocity profiles for more accurate fit
            # deleteat!(Global_Horz_profile, Trim_edges_horz)
            # deleteat!(Global_Vert_profile, Trim_edges_vert)
            crop_Z_plot_fine = convert(Array{Float64,1}, crop_Z_plot_fine)
            crop_Y_plot_fine = convert(Array{Float64,1}, crop_Y_plot_fine)
            # deleteat!(crop_Z_plot_fine, Trim_edges_vert)
            # deleteat!(crop_Y_plot_fine, Trim_edges_horz)

            Horz_profile_fine = Global_Horz_profile

            """Do Gaussian Fit"""
            # use trimmed profiles, no need to find max, just use max found earlier

            # Prepare mirrored gaussian profiles to better match portion of wake that will be
            # running into downstream turbines

            # Using moving average to smooth out Vertical Profile
            Vert_prof_previous = Global_Vert_profile
            Vert_prof_previous_fine = Global_Vert_profile
            Vert_profile_fine = Global_Vert_profile
            print(size(crop_Z_plot_fine), "\n")
            print(size(Vert_profile_fine), "\n")

            # Interpolate Vert_profile first (an initial smoothing)
            interp_Z_spline = CubicSpline(crop_Z_plot_fine, Vert_profile_fine)
            ZZ = minimum(crop_Z_plot_fine):0.005:maximum(crop_Z_plot_fine)
            interp_Z_profile = interp_Z_spline(ZZ)

            ZZ_smooth = ZZ
            # interp_Z_profile = movingaverage(interp_Z_profile, avg_factor)

            # # This smoothes out spikes effeciently, but it can also unrealistically lower peak velocity deficit

            Max_loc_Z = findmax(interp_Z_profile)
            Z_cm_comp = ZZ[Max_loc_Z[2]]

            """Upper Profile"""
            ZZ_upper = ZZ[Max_loc_Z[2]:end]
            Upper_Z_profile = interp_Z_profile[Max_loc_Z[2]:end]

            # flip it and append it to make a whole velocity profile
            Lower_Z_profile = reverse(Upper_Z_profile)
            Interp_Z_profile = vcat(Lower_Z_profile, Upper_Z_profile)

            # Adjust ZZ to match Interp_Z_profile
            # create array of acending values from 1 to length(ZZ_upper)
            Adjust_array = 0.005:0.01:length(Lower_Z_profile)*0.01
            ZZ_lower = reverse(ZZ_upper.-Adjust_array)
            ZZ = vcat(ZZ_lower, ZZ_upper)
            plot_upper = findmax(Interp_Z_profile)

            # for plotting 
            Interp_Z_upper_plot = Interp_Z_profile[plot_upper[2]:end]
            ZZ_upper_plot = ZZ[plot_upper[2]:end]

            # now take gaussian fit of it.
            # print("size ZZ: ", size(ZZ), "\n")
            # global parameters = Param_val(ZZ[12870:end-12870], Interp_Z_profile[12870:end-12870])
            # global parameters = Param_val(ZZ, Interp_Z_profile)
            # Gfit_vert = fit(Normal, ZZ_ground, Interp_Z_profile_ground)
            # Gfit_vert = fit_gauss(parameters)       # manual fit
            # Z_val_upper = -200:0.01:300
            # Gfit_array_Z_upper = gaussvalues(Gfit_vert["σ"], Gfit_vert["μ"], Gfit_vert["A"], Z_val_upper)
            # sigma_z_d_comp = (Gfit_vert["σ"])/D
            

            # gaussian fit
            # Gfit_vert = fit(Normal, crop_Z_plot, Vert_profile)
            Gfit_vert = fit(Normal, ZZ, Interp_Z_profile)
            Z_val_upper = -200:0.01:300
            Gfit_array_Z_upper = pdf(Gfit_vert, Z_val_upper)
            sigma_z_d_comp = (Gfit_vert.σ)/D
            # print("upper: ", Gfit_vert.μ, "\n")


            addition = 0.0
            # solve for constant to multiply gfit_array_Z_upper by 
            max_c = findmax(Gfit_array_Z_upper)
            Max_def_z = Gfit_array_Z_upper[max_c[2]]
            multiply_c = (Max_loc_Z[1]+addition)/Max_def_z

            # Gfit_array_Z_upper = Gfit_array_Z_upper*Gfit_vert["A"]
            Gfit_array_Z_upper = Gfit_array_Z_upper
            Gfit_array_Z_upper_plot = Gfit_array_Z_upper[max_c[2]:end]*multiply_c
            Gfit_array_Z_upper_plot_full = Gfit_array_Z_upper*multiply_c
            Z_val_upper_plot = Z_val_upper[max_c[2]:end]


            """Lower Profile"""
            ZZ_lower = ZZ_smooth[1:Max_loc_Z[2]]
            Lower_Z_profile = interp_Z_profile[1:Max_loc_Z[2]]
            Upper_Z_profile = reverse(Lower_Z_profile)
            Interp_Z_profile_ground = vcat(Lower_Z_profile, Upper_Z_profile)

            # flip it and append it to make a whole velocity profile
            Adjust_array = 0.005:0.01:length(Lower_Z_profile)*0.01
            ZZ_higher = reverse(ZZ_lower.+reverse(Adjust_array))
            ZZ_ground = vcat(ZZ_lower, ZZ_higher)

            # for plotting
            plot_under = findmax(Interp_Z_profile_ground)
            Interp_Z_lower_plot = Interp_Z_profile_ground[1:plot_under[2]]
            ZZ_under_plot = ZZ_ground[1:plot_under[2]]

            """THIS FIT ISN'T PERFORMING WELL"""
            """NEED TO WRITE MANUAL SOLVE FOR SIGMA"""

            
            
            # global parameters = Param_val(ZZ_ground[12870:end-12870], Interp_Z_profile_ground[12870:end-12870])
            global parameters = Param_val(ZZ_ground, Interp_Z_profile_ground)
            Gfit_vert = fit(Normal, ZZ_ground, Interp_Z_profile_ground)
            # Gfit_vert = fit_gauss(parameters)       # manual fit
            
            Z_val_ground = -200:0.01:300
            Gfit_array_Z_ground = pdf(Gfit_vert, Z_val_ground)
            sigma_z_d_comp_ground = (Gfit_vert.σ)/D

            # Z_val_ground = -200:0.01:300
            # Gfit_array_Z_ground = gaussvalues(Gfit_vert["σ"], Gfit_vert["μ"], Gfit_vert["A"], Z_val_ground)
            # sigma_z_d_comp_ground = (Gfit_vert["σ"])/D
            

            # solve for constant to multiply gfit_array_Z_ground by 
            max_c = findmax(Gfit_array_Z_ground)
            Max_def_z = Gfit_array_Z_ground[max_c[2]]
            multiply_c_ground = (Max_loc_Z[1]+addition)/Max_def_z

            # Gfit_array_Z_ground = Gfit_array_Z_ground*Gfit_vert["A"]
            Gfit_array_Z_ground = Gfit_array_Z_ground
            Gfit_array_Z_ground_plot = Gfit_array_Z_ground[1:max_c[2]]*multiply_c_ground
            Z_val_ground_plot = Z_val_ground[1:max_c[2]]

            """Horizontal"""
            Gfit_horz = fit(Normal, crop_Y_plot_fine, Horz_profile_fine)
            Y_val = 2250:0.005:2750
            Max_loc_Y = findmax(Horz_profile_fine)
            Gfit_array_Y = pdf(Gfit_horz, Y_val)
            max_cy = findmax(Gfit_array_Y)
            Max_def_y = Gfit_array_Y[max_cy[2]]
            multiply_c_y = Max_loc_Y[1]/Max_def_y
            maxy = findmax(Gfit_array_Y)
            Y_cm_comp = Y_val[maxy[2]]
            sigma_y_d_comp = (Gfit_horz.σ)/D

            Gfit_array_Y = Gfit_array_Y*multiply_c_y

            # global parameters = Param_val(crop_Y_plot_fine, Horz_profile_fine)
            # # Gfit_vert = fit(Normal, ZZ_ground, Interp_Z_profile_ground)
            # Gfit_horz = fit_gaussy(parameters)       # manual fit
            # Y_val = 2250:0.005:2750
            # # Z_val_upper = -200:0.01:300ç
            # Gfit_array_Y = gaussvalues(Gfit_horz["σ"], Gfit_horz["μ"], Gfit_horz["A"], Y_val)
            # maxy = findmax(Gfit_array_Y)
            # Y_cm_comp = Y_val[maxy[2]]
            # sigma_y_d_comp = (Gfit_horz["σ"])/D

            
            # cropped_SOWFA_data_comp_neg = negative[:, :]
            # YZ_grid_comp = YZ_grid_comp[zn2:zn1, xn1-100:xn2-100]
            # crop_Y_plot_comp = Y_plot[xn1:xn2]
            # crop_Z_plot_comp = Y_plot[zn2:zn1] 

            fontsz = 13
            linew = 2
            marks = 5

            # Save figures of the gauss fits
            # plot(crop_Z_plot, Vert_profile, label="Vertical Profile")
            # plot(ZZ_upper_plot, Interp_Z_upper_plot, label="upper Interpolation")
            # plot!(ZZ_under_plot, Interp_Z_lower_plot, label="lower Interpolation")
            plot(crop_Z_plot_fine/90, Vert_prof_previous, color=:black,linestyle=:dot, label="SOWFA")
            # plot!(ZZ_smooth, interp_Z_profile, label="Smooth SOWFA")
            # plot(crop_Z_plot_fine/90, Vert_prof_previous, seriestype=:scatter, label="SOWFA")

            Z_val_plot = vcat(Z_val_ground_plot, Z_val_upper_plot)
            print("Z_val_plot: ", size(Z_val_plot))
            Gfit_array_Z_full = vcat(Gfit_array_Z_ground_plot, Gfit_array_Z_upper_plot)
            print("Gfit_array_Z_full: ", size(Gfit_array_Z_full))
            # plot!(Z_val_upper_plot/90, Gfit_array_Z_upper_plot, color=:dodgerblue, label="Piece-wise Gaussian Fit")
            plot!(Z_val_plot/90, Gfit_array_Z_full,color=:dodgerblue,linewidth=linew, xlimit=[0,3.5], label="Piece-wise Gaussian Fit", grid=:false, xlabel="z*", ylabel="Velocity Deficit (m/s)", xrotation = 0,foreground_color_legend = nothing,yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz)
            print("gauss_name_z: ", gauss_name_z)
            savefig(gauss_name_z)
            

            # plotting full gaussain fit for upper profile fit
            plot(crop_Z_plot_fine/90, Vert_prof_previous,linewidth=linew, color=:black,linestyle=:dot, label="SOWFA")
            plot!(Z_val_upper/90, Gfit_array_Z_upper_plot_full,linewidth=5, color=:dodgerblue3, label="Gaussian Fit", xlimit=[0,3.5], grid=:false, xlabel="z*", ylabel="Velocity Deficit (m/s)", xrotation = 0,foreground_color_legend = nothing,yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz)
            savefig(gauss_name_z_full)

            plot((crop_Y_plot_fine.-2500)/126, Horz_profile_fine,linewidth=linew, color=:black,linestyle=:dot, label="SOWFA")
            plot!((Y_val.-2500)/126, Gfit_array_Y,linewidth=5, label="Gaussian Fit", color=:dodgerblue3, grid=:false, xlabel="y/D", ylabel="Velocity Deficit (m/s)", xrotation = 0,foreground_color_legend = nothing,yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz)
            savefig(gauss_name_y)

            print("Y_cm: ", Y_cm_comp, "\n")
            print("Z_cm: ", Z_cm_comp, "\n")
            print("sig z: ", sigma_z_d_comp, "\n")
            print("sig y: ", sigma_y_d_comp, "\n")
            print("sigma z ground: ", sigma_z_d_comp_ground, "\n")
            sigmay[i,j] = sigma_y_d_comp
            sigmaz[i,j] = sigma_z_d_comp
            sigmaz_ground[i,j] = sigma_z_d_comp_ground
            defz[i,j] = Z_cm_comp
            defy[i,j] = Y_cm_comp
            cup[i,j] = multiply_c
            cdown[i,j] = multiply_c_ground
            power_downstream, avg_vel = power_estimate(cropped_SOWFA_data_comp_fine, 90.0, 2500.0, D, crop_Z_plot_fine, crop_Y_plot_fine, power_filename)
            power_upstream = power_estimate_tilt(cropped_SOWFA_data_ambient_fine,90.0, 2500.0, D, crop_Z_plot_fine, crop_Y_plot_fine, tilt, 1.88, power_filename_down)
            power[i,j] = power_downstream
            power_up[i,j] = power_upstream
            print("power_up: ", power_up, "\n")
            print("power_prediction: ", power_downstream, "\n")
            avg_V[i,j] = avg_vel


            print("y: ", size(crop_Y_plot_fine), "\n")
            print("z: ", size(crop_Z_plot_fine), "\n")
            print("z_first: ", crop_Z_plot_fine[1], "\n")
            print("data: ", size(cropped_SOWFA_data_comp_neg_fine), "\n")
            data1 = heatmap((crop_Y_plot_fine.-2500)/126, (crop_Z_plot_fine/90),cropped_SOWFA_data_comp_neg_fine, c=:Blues)
            # plot(data1, colorbar_title=" \nVelocity Deficit (m/s)",clim=(0,1.35), right_margin=5Plots.mm, legend=:false, grid=:false, xlabel="y/D", ylabel="z*", xrotation = 0)
            plot(data1, colorbar_title=" \nVelocity Deficit (m/s)",clim=(0,1.2), right_margin=5Plots.mm, grid=:false, xlabel="y/D", ylabel="z*")
            # plot(data1, colorbar_title=" \nVelocity Deficit (m/s)",clim=(0,0.75), right_margin=5Plots.mm, legend=:false, grid=:false, xlabel="y/D", ylabel="z*", xrotation = 0)

            # plot(data1, colorbar_title="Velocity Deficit (m/s)",clim=(0,0.75), legend=:false, grid=:false, xlabel="y/D", ylabel="z*", xrotation = 0)
            # # plot!([crop_Y_plot[min_loc[1]]], [crop_Z_plot[min_loc[2]]], color="blue")
            # # scatter!([crop_Y_plot[min_loc[2]]], [crop_Z_plot[min_loc[1]]], color="blue", label="wake center")
            f(t) = ((126/2).*cos.(t))/126
            g(t) = (90 .+ (126/2).*sin.(t))/90
            range = 0:0.01:2*pi
            x_values = f(range)
            y_values = g(range)

            # # scatter!((crop_Y_plot_comp[Y_whole_wake_comp].-2500)/126, (crop_Z_plot_comp[X_whole_wake_comp])/90, color="red", label="Wake")
            scatter!([0.0], [1.0],markerstrokewidth=0, markersize = marks, color="firebrick", label="")
            # # scatter!(([Y_cm_comp_threshold].-2500)/126, [Z_cm_comp_threshold]/90, color="brown", label="Center based on max Deficit")
            # # scatter!(([Y_cm_comp_max].-2500)/126, [Z_cm_comp_max]/90, color="magenta", label="Center based on max Deficit")
            scatter!(([Y_cm_comp].-2500)/126, [Z_cm_comp]/90,markerstrokewidth=0, markersize = marks, color="green", label="Wake Center")
            plot!(x_values, y_values, label="Rotor",linewidth=linew+1,legend=:false, color="firebrick", xlabel="y/D", ylabel="z*", xrotation = 0)

            xlims!((-2,2))
            ylims!((0,3.388888889))
            # plot!(size=((14/8)*400,400),yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz,foreground_color_legend = nothing,)

            plot!(size=((14/8)*400,400),yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz,foreground_color_legend = nothing)
            savefig(moviename)

        end
    end

    return sigmay, sigmaz, defz, defy, sigmaz_ground, cup, cdown, power, power_up, avg_V
end

function power_estimate(velocity_slice, hub_height, Y, D, Z_array, Y_array, power_filename)
    counter = 0
    Velocity_Total = 0
    # i represents Z ?
    # j represents Y?
    # print("length(velocity_slice[:,1]: ", length(velocity_slice[:,1]), "\n")
    # print("length(velocity_slice[:,1]: ", length(velocity_slice[1,:]), "\n")
    # print("Z_array: ", length(Z_array), "\n")
    # print("Y_array: ", length(Y_array), "\n")

    for i in 1:length(velocity_slice[:,1])
        for j in 1:length(velocity_slice[1,:])
            posZ = Z_array[i]
            posY = Y_array[j]
            diffZ = abs(posZ - hub_height)
            # print("diffZ: ", diffZ, "\n")
            
            diffY = abs(posY - Y)
            # print("diffY: ", diffY, "\n")
            distance = sqrt(diffZ^2 + diffY^2)
            # print("distance: ", distance, "\n")
            
            if distance < D/2

                # inside of rotor swept Area
                counter = counter + 1
                Velocity_Total = Velocity_Total + velocity_slice[i,j]
            end
        end
    end

    Average_velocity = Velocity_Total/counter
    print("avg V: ", Average_velocity, "\n")
    A = pi*(D/2)^2
    rho = 1.1716    # kg/m^3
    Cp = 4.631607703567137690e-01

    fontsz = 13
    linew = 2
    marks = 5

    # Power calculation
    Power = (0.5*rho*A*Average_velocity^3)*Cp

    # data1 = heatmap((Y_array.-2500)/126, (Z_array/90),velocity_slice, c=:Blues)
    # # plot(data1, colorbar_title=" \nVelocity Deficit (m/s)",clim=(0,1.35), right_margin=5Plots.mm, legend=:false, grid=:false, xlabel="y/D", ylabel="z*", xrotation = 0)
    # plot(data1, colorbar_title=" \nVelocity (m/s)",clim=(3,9), right_margin=5Plots.mm, grid=:false, xlabel="y/D", ylabel="z*", xrotation = 0)
    # f(t) = ((126/2).*cos.(t))/126
    # g(t) = (90 .+ (126/2).*sin.(t))/90
    # range = 0:0.01:2*pi
    # x_values = f(range)
    # y_values = g(range)
    # plot!(x_values, y_values, label="Rotor",linewidth=linew+1,legend=:false, color="firebrick", xlabel="y/D", ylabel="z*", xrotation = 0)

    # scatter!([0.0], [1.0],markerstrokewidth=0, markersize = marks, color="firebrick", label="")
    # savefig(power_filename)
    return Power, Average_velocity

end


function power_estimate_tilt(velocity_slice, hub_height, Y, D, Z_array, Y_array, tilt, pP, power_filename_down)
    counter = 0
    Velocity_Total = 0
    # i represents Z ?
    # j represents Y?
    # print("length(velocity_slice[:,1]: ", length(velocity_slice[:,1]), "\n")
    # print("length(velocity_slice[:,1]: ", length(velocity_slice[1,:]), "\n")
    # print("Z_array: ", length(Z_array), "\n")
    # print("Y_array: ", length(Y_array), "\n")

    # for i in 1:length(velocity_slice[:,1])
    #     for j in 1:length(velocity_slice[1,:])
    #         posZ = Z_array[i]
    #         posY = Y_array[j]
    #         diffZ = abs(posZ - hub_height)
    #         # print("diffZ: ", diffZ, "\n")
            
    #         diffY = abs(posY - Y)
    #         # print("diffY: ", diffY, "\n")
    #         distance = sqrt(diffZ^2 + diffY^2)
    #         # print("distance: ", distance, "\n")
            
    #         if distance < D/2

    #             # inside of rotor swept Area
    #             counter = counter + 1
    #             Velocity_Total = Velocity_Total + velocity_slice[i,j]
    #         end
    #     end
    # end

    # Average_velocity = Velocity_Total/counter
    Average_velocity = 8.1
    print("avg V: ", Average_velocity, "\n")
    A = pi*(D/2)^2
    rho = 1.1716    # kg/m^3
    Cp = 4.631607703567137690e-01

    # Power calculation
    Power = (0.5*rho*A*Average_velocity^3)*Cp*cos(tilt)^(pP)

    # data1 = heatmap((Y_array.-2500)/126, (Z_array/90),velocity_slice, c=:Blues)
    # # plot(data1, colorbar_title=" \nVelocity Deficit (m/s)",clim=(0,1.35), right_margin=5Plots.mm, legend=:false, grid=:false, xlabel="y/D", ylabel="z*", xrotation = 0)
    # plot(data1, colorbar_title=" \nVelocity (m/s)", right_margin=5Plots.mm, grid=:false, xlabel="y/D", ylabel="z*", xrotation = 0)
    # savefig(power_filename_down)

    return Power

end

function vel_deficit(tilt, X_D, def_z, C_T, sigy, sigz)
    # determine where we are looking at velocity deficit (@ X_D)
    # parameters:
    wakesave = join([string(tilt), "_", string(X_D), ".pdf"])
    
    def_z = def_z-90.0
    print("tilt: ", tilt, "\n")
    print("X_D: ", X_D, "\n")
    print("def_z: ", def_z, "\n")
    print("C_T: ", C_T, "\n")
    D = 126.0
    # sample randomly from within downstream rotor swept areas
    # create x and y values within swept area
    Y_values = (rand(5000).-0.5)*D
    Z_values = (rand(5000).-0.5)*D
    delete_index = []

    for i = 1:length(Y_values)
        diffZ = Y_values[i]
        diffY = Z_values[i]
        distance = sqrt(diffZ^2 + diffY^2)
        
        if distance > D/2
            push!(delete_index, i)
        end

    end
    deleteat!(Y_values, delete_index)
    deleteat!(Z_values, delete_index)
    # vel_deficit(tilt[1], 6.0, defZ_neg_5[index], C_T)

    # find vel_deficit at each of the points
    vel_deficit = @. vel_deficit_calculations(C_T, tilt, X_D, D, Y_values, Z_values, def_z, sigy, sigz)
    data1 = surface(Y_values, Z_values, vel_deficit)
    plot(data1,  camera=(0, 90), aspect_ratio=:equal)
    savefig(wakesave)
    # average results and return
    Vel_average = sum(vel_deficit)/length(Y_values)

    return Vel_average
end

function vel_deficit_calculations(C_T, tilt, X_D, D, y_val, z_val, deflection, sigy_D, sigz_D)
    sigy_D, sigz_D = sigysigz(X_D, tilt)
    sigy = sigy_D*D
    sigz = sigz_D*D
    # print("sigy: ", sigy, "\n")
    # print("sigz: ", sigz, "\n")
    b = (C_T*cos(tilt))/(8*(sigy_D*sigz_D))
    c = exp(-0.5*(y_val/sigy)^2)
    d = exp(-0.5*((z_val-deflection)/sigz)^2)
    vel_deficit_value = (1 - sqrt(1 - b))*c*d
    return vel_deficit_value
end

function sigysigz(X_D, tilt)
    if tilt < 0
        kz = 0.025
        ky = 0.017
        sigy = ky*X_D + 0.266
        sigz = kz*X_D + cos(tilt)*0.266
    else
        kz = -0.563*tilt^2 + 0.108*tilt + 0.027
        ky = 0.048*tilt + 0.018
        sigzo = 0.168 - 0.014*log(tilt-0.0419)
        # sigzo = 0.20
        sigyo = 0.266

        sigy = (ky*X_D) + sigyo
        sigz = (kz*X_D) + sigzo
    end

    return sigy, sigz
end

# function sigysigz(X_D, tilt)

#     kz = -0.563*tilt^2 + 0.108*tilt + 0.027
#     ky = 0.048*tilt + 0.018
#     sigzo = 0.168 - 0.014*log(tilt-0.0419)
#     # sigzo = 0.20
#     sigyo = 0.266

#     sigy = (ky*X_D) + sigyo
#     sigz = (kz*X_D) + sigzo


#     return sigy, sigz
# end

function SOWFA_multiple(movnames, files, filenames, tilt)
    for i in 1:length(files)
        file_1 = files[i]
        filename = filenames[i]
        movnam = movnames[i]

        # slice = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        D = 126
        # slice = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        # slices = 6:0.2:15
        slices = 7.0:1.0:12.0
        # slices = 7:0.2:7.1

        # for power plotting
        # slices = [6.0, 8.0, 10.0, 12.0]
        # slices = [6.0]
        # smoothing = [1000, 3000, 5000, 7000, 9000, 11000]
        # smoothing = 3000:500:8000
        
        # 
        # smoothing = 3000
        # smooth
        smoothing = 5000
        # normal
        # smoothing = 100
        # smoothing = 3000
        # slice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        D = 126.0 
        # movnam = "2.5"
        tol_comp = 0.7
        # threshold = 0.5:0.1:1.3
        threshold = 0.7

        sigmay_ht = zeros(length(slices), length(smoothing))
        sigmaz_ht = zeros(length(slices), length(smoothing))
        defz_ht = zeros(length(slices), length(smoothing))
        defy_ht = zeros(length(slices), length(smoothing))
        sigmaz_g = zeros(length(slices), length(smoothing))
        power_t = zeros(length(slices), length(smoothing))
        power_up = zeros(length(slices), length(smoothing))
        avgV_t = zeros(length(slices), length(smoothing))



        moviename = join([movnam])
        gauss_name_z = join([movnam, "_gauss_z"])
        gauss_name_y = join([movnam, "_gauss_y_new"])
        print(moviename)
        sigy, sigmz, dz, dy, sigmz_g, cup, cdown, power_val, power_up_val, avgV_val = SOWFA_analysis(slices, moviename, gauss_name_z, gauss_name_y, threshold, file_1, smoothing, tilt[i])
        sigmay_ht[:,:] = sigy
        sigmaz_ht[:,:] = sigmz
        defz_ht[:,:] = dz
        defy_ht[:,:] = dy
        sigmaz_g[:,:] = sigmz_g
        power_t[:,:] = power_val
        power_up[:,:] = power_up_val
        avgV_t[:,:] = avgV_val
        print(power_t)

        save(filename, "sigy", sigmay_ht, "sigz", sigmaz_ht, "defz", defz_ht, "defy", defy_ht, "sigz_g", sigmaz_g, "cup", cup, "cdown", cdown, "power", power_t, "power_up", power_up_val, "avgV", avgV_t)
    end
end

function deflection_est(tilt, x_D)
    c1 = 2.0921
    c2 = -7.9725
    c3 = -0.0854
    c4 = 0.0041
    c5 = -.3663
    c6 = 0.9701
    c7 = 0.0045
    c8 = 0.2840

    defz = @. (c1*tilt) + (c2*tilt^2) + (c3*x_D) + (c4*(x_D)^2) + (c5*x_D*tilt) + (c6*(tilt^2)*x_D) + (c7*(x_D^2)*tilt) + c8
    return defz
end

# final Data Values
file_0 = "/Users/jamescutler/Downloads/byu_tilt_runs_single/c_000_sp7_1turb_hNormal_D126_tilt_base/lite_data_000.ftr"
file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_001_sp7_1turb_hNormal_D126_tilt_2.5/lite_data_001.ftr"
file_2 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_004_sp7_1turb_hNormal_D126_tilt_5/lite_data_004.ftr"
file_3 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_002_sp7_1turb_hNormal_D126_tilt_7.5/lite_data_002.ftr"
file_4 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_005_sp7_1turb_hNormal_D126_tilt_10/lite_data_005.ftr"
file_5 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_003_sp7_1turb_hNormal_D126_tilt_12.5/lite_data_003.ftr"

# Final data values
filename0 = "analysis_smoothing_piece_wise_largeest_span_base_normal_n_finall.jld"
filename1 = "analysis_smoothing_piece_wise_largeest_span_2.5_normal_n_finall.jld"
filename2 = "analysis_smoothing_piece_wise_largeest_span_5_normal_n_finall.jld"
filename3 = "analysis_smoothing_piece_wise_largeest_span_7.5_normal_n_finall.jld"
filename4 = "analysis_smoothing_piece_wise_largeest_span_10_normal_n_finall.jld"
filename5 = "analysis_smoothing_piece_wise_largeest_span_12.5_normal_n_finall.jld"


# Files for figures
files = [file_0, file_1, file_2, file_3, file_4, file_5]
movnames = ["neg5", "2.5", "5", "7.5", "10", "12.5"]
filenames = [filename0, filename1, filename2, filename3, filename4, filename5]
tilt = [-5.0, 2.5, 5.0, 7.5, 10.0, 12.5] * pi/180.0

# tilt = [2.5] * pi/180.0
# SOWFA_multiple(movnames, files, filenames, tilt)
D = 126
slices = 7.0:0.2:12.0
fontsz = 13
linew = 2
marks = 5

sigmaY_neg_5 = load(filename0, "sigy")
sigmaZ_neg_5 = load(filename0, "sigz")
defZ_neg_5 = load(filename0, "defz")
defY_neg_5 = load(filename0, "defy")
sigmaZ_g_neg_5 = load(filename0, "sigz_g")
cup_neg_5 = load(filename0, "cup")
cdown_neg_5 = load(filename0, "cdown")
power_neg_5 = load(filename0, "power")
power_neg_5_up = load(filename0, "power_up")
avgV_neg_5 = load(filename0, "avgV")


sigmaY_2_5 = load(filename1, "sigy")
sigmaZ_2_5 = load(filename1, "sigz")
defZ_2_5 = load(filename1, "defz")
defY_2_5 = load(filename1, "defy")
sigmaZ_g_2_5 = load(filename1, "sigz_g")
cup_2_5 = load(filename1, "cup")
cdown_2_5 = load(filename1, "cdown")
power_2_5 = load(filename1, "power")
power_2_5_up = load(filename1, "power_up")
avgV_2_5 = load(filename1, "avgV")


sigmaY_5 = load(filename2, "sigy")
sigmaZ_5 = load(filename2, "sigz")
defZ_5 = load(filename2, "defz")
defY_5 = load(filename2, "defy")
sigmaZ_g_5 = load(filename2, "sigz_g")
cup_5 = load(filename2, "cup")
cdown_5 = load(filename2, "cdown")
power_5 = load(filename2, "power")
power_5_up = load(filename2, "power_up")
avgV_5 = load(filename2, "avgV")


sigmaY_7_5 = load(filename3, "sigy")
sigmaZ_7_5 = load(filename3, "sigz")
defZ_7_5 = load(filename3, "defz")
defY_7_5 = load(filename3, "defy")
sigmaZ_g_7_5 = load(filename3, "sigz_g")
cup_7_5 = load(filename3, "cup")
cdown_7_5 = load(filename3, "cdown")
power_7_5 = load(filename3, "power")
power_7_5_up = load(filename3, "power_up")
avgV_7_5 = load(filename3, "avgV")


sigmaY_10 = load(filename4, "sigy")
sigmaZ_10 = load(filename4, "sigz")
defZ_10 = load(filename4, "defz")
defY_10 = load(filename4, "defy")
sigmaZ_g_10 = load(filename4, "sigz_g")
cup_10 = load(filename4, "cup")
cdown_10 = load(filename4, "cdown")
power_10 = load(filename4, "power")
power_10_up = load(filename4, "power_up")
avgV_10 = load(filename4, "avgV")


sigmaY_12_5 = load(filename5, "sigy")
sigmaZ_12_5 = load(filename5, "sigz")
defZ_12_5 = load(filename5, "defz")
defY_12_5 = load(filename5, "defy")
sigmaZ_g_12_5 = load(filename5, "sigz_g")
cup_12_5 = load(filename5, "cup")
cdown_12_5 = load(filename5, "cdown")
power_12_5 = load(filename5, "power")
power_12_5_up = load(filename5, "power_up")
avgV_12_5 = load(filename5, "avgV")

# Set FONTS
font25_c = text("", 12, color=:lightskyblue2).font
font5_c = text("", 12,color=:skyblue2).font
font75_c = text("", 12,color=:steelblue2).font
font10_c = text("", 12,color=:dodgerblue).font
font125_c = text("", 12,color=:dodgerblue3).font
fontylabel = text("", 12,color=:black).font

Y_ax = -0.19
Z_fix = -5


"""KY FIT"""

# 2.5 tilt
Q = polyfit(slices, sigmaY_2_5, 1)
ky_val_2_5 = Q[1]
eps_y_2_5 = Q[0]
Y_values_2_5 = eps_y_2_5.+ky_val_2_5.*slices[1:end]

ky_val_2_5 = round(ky_val_2_5, digits=3)

# 5 tilt
Q = polyfit(slices, sigmaY_5, 1)
ky_val_5 = Q[1]
eps_y_5 = Q[0]
Y_values_5 = eps_y_5.+ky_val_5.*slices[1:end]

ky_val_5 = round(ky_val_5, digits=3)

# 7.5 tilt
Q = polyfit(slices, sigmaY_7_5, 1)
ky_val_7_5 = Q[1]
eps_y_7_5 = Q[0]
Y_values_7_5 = eps_y_7_5.+ky_val_7_5.*slices[1:end]

ky_val_7_5 = round(ky_val_7_5, digits=3)

# 10 tilt
Q = polyfit(slices, sigmaY_10, 1)
ky_val_10 = Q[1]
eps_y_10 = Q[0]
Y_values_10 = eps_y_10.+ky_val_10.*slices[1:end]

ky_val_10 = round(ky_val_10, digits=3)

# 12.5 tilt
Q = polyfit(slices, sigmaY_12_5, 1)
ky_val_12_5 = Q[1]
eps_y_12_5 = Q[0]
Y_values_12_5 = eps_y_12_5.+ky_val_12_5.*slices[1:end]

ky_val_12_5 = round(ky_val_12_5, digits=3)

X_ax = 12.35


# # # Plot
X_ax = 12.4
X_ax1 = 6.65
X_ax2 = 6.2
Y_ax1 = 0.338
Xline = [6, 6.4, 6.4]
Y_2_5 = [Y_values_2_5[1], Y_values_2_5[1], Y_values_2_5[3]]
Y_5 = [Y_values_5[1], Y_values_5[1], Y_values_5[3]]
Y_7_5 = [Y_values_7_5[1], Y_values_7_5[1], Y_values_7_5[3]]
Y_10 = [Y_values_10[1], Y_values_10[1], Y_values_10[3]]
Y_12_5 = [Y_values_12_5[1], Y_values_12_5[1], Y_values_12_5[3]]

# Rotating annotations
font25 = text("", 9).font
font25.rotation=25
font5 = text("", 9).font
font5.rotation=25
font75 = text("", 9).font
font75.rotation=25
font10 = text("", 9).font
font10.rotation=27
font125 = text("", 9).font
font125.rotation=29



# Coloring the tilt values
font25_c = text("", 12, color=:lightskyblue2).font
font5_c = text("", 12,color=:skyblue2).font
font75_c = text("", 12,color=:steelblue2).font
font10_c = text("", 12,color=:dodgerblue).font
font125_c = text("", 12,color=:dodgerblue3).font
fontylabel = text("", 12,color=:black).font
Y_ax = 4.95
Z_fix = 0.005

# # # Plot
plot(slices, sigmaY_2_5, seriestype=:scatter,color=:lightskyblue2, label="2.5°")
plot!(slices, Y_values_2_5,color=:lightskyblue2, label="ky = $ky_val_2_5")
plot!(slices, sigmaY_5, seriestype=:scatter,color=:skyblue2, label="5°")
plot!(slices, Y_values_5,color=:skyblue2, label="ky = $ky_val_5")
plot!(slices, sigmaY_7_5, seriestype=:scatter,color=:steelblue2, label="7.5°")
plot!(slices, Y_values_7_5,color=:steelblue2, label="ky = $ky_val_7_5")
plot!(slices, sigmaY_10, seriestype=:scatter,color=:dodgerblue,label="10°")
plot!(slices, Y_values_10,color=:dodgerblue, label="ky = $ky_val_10")
plot!(slices, sigmaY_12_5, seriestype=:scatter,color=:dodgerblue3, label="12.5°")
plot!(slices, Y_values_12_5,color=:dodgerblue3, label="ky = $ky_val_12_5", xlabel="x/D (Downstream Distance)", xrotation = 0,legend=:false,xlimit = [6, 12.5], legendfontsize=10, grid=false, xticks=[6, 7, 8, 9, 10, 11, 12], dpi=300)
plot!(annotations = ([X_ax1], Y_values_2_5[3]+Z_fix-(Y_values_2_5[3]-Y_values_2_5[1])/2, text("$ky_val_2_5", font25)))
plot!(Xline, Y_2_5, color=:lightskyblue2)
plot!(annotations = ([X_ax1], Y_values_5[3]+Z_fix-0.001-(Y_values_5[3]-Y_values_5[1])/2, text("$ky_val_5", font5)))
plot!(Xline, Y_5, color=:skyblue2)
plot!(annotations = ([X_ax1], Y_values_7_5[3]+Z_fix+0.001-(Y_values_7_5[3]-Y_values_7_5[1])/2, text("$ky_val_7_5", font75)))
plot!(Xline, Y_7_5, color=:steelblue2)
plot!(annotations = ([X_ax1], Y_values_10[3]+Z_fix+0.002-(Y_values_10[3]-Y_values_10[1])/2, text("$ky_val_10", font10)))
plot!(Xline, Y_10, color=:dodgerblue)
plot!(annotations = ([X_ax1], Y_values_12_5[3]+Z_fix+0.002-(Y_values_12_5[3]-Y_values_12_5[1])/2, text("$ky_val_12_5", font125)))
plot!(Xline, Y_12_5, color=:dodgerblue3)
plot!(annotations = ([X_ax2], [Y_ax1], "1", 9))
plot!(annotations = ([Y_ax], 0.505,text("{\\sigma_y/D}", fontylabel)))
plot!(annotations = ([X_ax], sigmaY_2_5[end-M],text("2.5°", font25_c)))
plot!(annotations = ([X_ax], sigmaY_5[end-M], text("5.0°", font5_c)))
plot!(annotations = ([X_ax], sigmaY_7_5[end-M], text("7.5°", font75_c)))
plot!(annotations = ([X_ax], sigmaY_10[end-M], text("10.0°", font10_c)))
plot!(annotations = ([X_ax], sigmaY_12_5[end-M], text("12.5°", font125_c)), left_margin=7Plots.mm, transparent=true, tickfontsize=12, guidefontsize=12)
savefig("sigma_y_fit.pdf")












"""KZ FIT UPPER"""

# 2.5 tilt
Q = polyfit(slices, sigmaZ_2_5, 1)
kz_val_2_5 = Q[1]
eps_z_2_5 = Q[0]
Z_values_2_5 = eps_z_2_5.+kz_val_2_5.*slices

kz_val_2_5 = round(kz_val_2_5, digits=3)

# 5 tilt
Q = polyfit(slices, sigmaZ_5, 1)
kz_val_5 = Q[1]
eps_z_5 = Q[0]
Z_values_5 = eps_z_5.+kz_val_5.*slices

kz_val_5 = round(kz_val_5, digits=3)

# 7.5 tilt
Q = polyfit(slices, sigmaZ_7_5, 1)
kz_val_7_5 = Q[1]
eps_z_7_5 = Q[0]
Z_values_7_5 = eps_z_7_5.+kz_val_7_5.*slices

kz_val_7_5 = round(kz_val_7_5, digits=3)

# 10 tilt
Q = polyfit(slices, sigmaZ_10, 1)
kz_val_10 = Q[1]
eps_z_10 = Q[0]
Z_values_10 = eps_z_10.+kz_val_10.*slices

kz_val_10 = round(kz_val_10, digits=3)

# 12.5 tilt
Q = polyfit(slices, sigmaZ_12_5, 1)
kz_val_12_5 = Q[1]
eps_z_12_5 = Q[0]
Z_values_12_5 = eps_z_12_5.+kz_val_12_5.*slices

kz_val_12_5 = round(kz_val_12_5, digits=3)


# # # Plot
X_ax = 12.4
X_ax1 = 6.65
X_ax2 = 6.2
Y_ax1 = 0.338
Xline = [6, 6.4, 6.4]
Y_2_5 = [Z_values_2_5[1], Z_values_2_5[1], Z_values_2_5[3]]
Y_5 = [Z_values_5[1], Z_values_5[1], Z_values_5[3]]
Y_7_5 = [Z_values_7_5[1], Z_values_7_5[1], Z_values_7_5[3]]
Y_10 = [Z_values_10[1], Z_values_10[1], Z_values_10[3]]
Y_12_5 = [Z_values_12_5[1], Z_values_12_5[1], Z_values_12_5[3]]

# Rotating annotations
font25 = text("", 9).font
font25.rotation=27
font5 = text("", 9).font
font5.rotation=27
font75 = text("", 9).font
font75.rotation=27
font10 = text("", 9).font
font10.rotation=25
font125 = text("", 9).font
font125.rotation=22

# Coloring the tilt values
font25_c = text("", 12, color=:lightskyblue2).font
font5_c = text("", 12,color=:skyblue2).font
font75_c = text("", 12,color=:steelblue2).font
font10_c = text("", 12,color=:dodgerblue).font
font125_c = text("", 12,color=:dodgerblue3).font
Z_fix = 0.005


plot(slices, sigmaZ_2_5, seriestype=:scatter,color=:lightskyblue2, label="2.5°")
plot!(slices, Z_values_2_5,color=:lightskyblue2, label="kz = $kz_val_2_5")
plot!(slices, sigmaZ_5, seriestype=:scatter,color=:skyblue2, label="5.0°")
plot!(slices, Z_values_5,color=:skyblue2, label="kz = $kz_val_5")
plot!(slices, sigmaZ_7_5, seriestype=:scatter,color=:steelblue2, label="7.5°")
plot!(slices, Z_values_7_5,color=:steelblue2, label="kz = $kz_val_7_5")
plot!(slices, sigmaZ_10, seriestype=:scatter,color=:dodgerblue,label="10.0°")
plot!(slices, Z_values_10,color=:dodgerblue, label="kz = $kz_val_10")
plot!(slices, sigmaZ_12_5, seriestype=:scatter,color=:dodgerblue3, label="12.5°")
plot!(slices, Z_values_12_5,color=:dodgerblue3, label="kz = $kz_val_12_5", xlabel="x/D (Downstream Distance)", ylimits=[0.34, 0.61], xlimits=[6.0,12.5], xrotation = 0,legend=:false, legendfontsize=10, grid=false, dpi=300)
# annotate!([X_ax1], Z_values_2_5[3]-(Z_values_2_5[3]-Z_values_2_5[1])/2, text("$kz_val_2_5"), annotationrotation=[45])
plot!(annotations = ([X_ax1], Z_values_2_5[3]+Z_fix-(Z_values_2_5[3]-Z_values_2_5[1])/2, text("$kz_val_2_5", font25)))
plot!(Xline, Y_2_5, color=:lightskyblue2)
plot!(annotations = ([X_ax1], Z_values_5[3]+Z_fix+0.002-(Z_values_5[3]-Z_values_5[1])/2, text("$kz_val_5", font5)))
plot!(Xline, Y_5, color=:skyblue2)
plot!(annotations = ([X_ax1], Z_values_7_5[3]+Z_fix-(Z_values_7_5[3]-Z_values_7_5[1])/2, text("$kz_val_7_5", font75)))
plot!(Xline, Y_7_5, color=:steelblue2)
plot!(annotations = ([X_ax1], Z_values_10[3]+Z_fix-(Z_values_10[3]-Z_values_10[1])/2, text("$kz_val_10", font10)))
plot!(Xline, Y_10, color=:dodgerblue)
plot!(annotations = ([X_ax1], Z_values_12_5[3]+Z_fix-0.001-(Z_values_12_5[3]-Z_values_12_5[1])/2, text("$kz_val_12_5", font125)))
plot!(Xline, Y_12_5, color=:dodgerblue3)
plot!(annotations = ([X_ax2], [Y_ax1], "1", 9))
plot!(annotations = ([Y_ax], 0.47,text("{\\sigma_z/D}", fontylabel)))
plot!(annotations = ([X_ax+0.09], sigmaZ_2_5[end-M],text("2.5°", font25_c)))
plot!(annotations = ([X_ax+0.09], sigmaZ_5[end-M], text("5.0°", font5_c)))
plot!(annotations = ([X_ax+0.09], sigmaZ_7_5[end-M], text("7.5°", font75_c)))
plot!(annotations = ([X_ax], sigmaZ_10[end-M], text("10.0°", font10_c)))
plot!(annotations = ([X_ax], sigmaZ_12_5[end-M], text("12.5°", font125_c)), left_margin=7Plots.mm, transparent=true, tickfontsize=12, guidefontsize=12)
savefig("sigma_z_upper_fit.pdf")




"""KZ FIT LOWER"""

# 2.5 tilt
Q = polyfit(slices, sigmaZ_g_2_5, 1)
kz_val_g_2_5 = Q[1]
eps_z_g_2_5 = Q[0]
Z_values_g_2_5 = eps_z_g_2_5.+kz_val_g_2_5.*slices

kz_val_g_2_5 = round(kz_val_g_2_5, digits=4)

# 5 tilt
Q = polyfit(slices, sigmaZ_g_5, 1)
kz_val_g_5 = Q[1]
eps_z_g_5 = Q[0]
Z_values_g_5 = eps_z_g_5.+kz_val_g_5.*slices

kz_val_g_5 = round(kz_val_g_5, digits=4)

# 7.5 tilt
Q = polyfit(slices, sigmaZ_g_7_5, 1)
kz_val_g_7_5 = Q[1]
eps_z_g_7_5 = Q[0]
Z_values_g_7_5 = eps_z_g_7_5.+kz_val_g_7_5.*slices

kz_val_g_7_5 = round(kz_val_g_7_5, digits=4)

# 10 tilt
Q = polyfit(slices, sigmaZ_g_10, 1)
kz_val_g_10 = Q[1]
eps_z_g_10 = Q[0]
Z_values_g_10 = eps_z_g_10.+kz_val_g_10.*slices

kz_val_g_10 = round(kz_val_g_10, digits=4)

# 12.5 tilt
Q = polyfit(slices, sigmaZ_g_12_5, 1)
kz_val_g_12_5 = Q[1]
eps_z_g_12_5 = Q[0]
Z_values_g_12_5 = eps_z_g_12_5.+kz_val_g_12_5.*slices

kz_val_g_12_5 = round(kz_val_g_12_5, digits=4)

# # # Plot
plot(slices, sigmaZ_g_2_5, seriestype=:scatter,color=:blue, label="2.5°")
plot!(slices, Z_values_g_2_5,color=:blue, label="kz = $kz_val_g_2_5")
plot!(slices, sigmaZ_g_5, seriestype=:scatter,color=:red, label="5°")
plot!(slices, Z_values_g_5,color=:red, label="kz = $kz_val_g_5")
plot!(slices, sigmaZ_g_7_5, seriestype=:scatter,color=:black, label="7.5°")
plot!(slices, Z_values_g_7_5,color=:black, label="kz = $kz_val_g_7_5")
plot!(slices, sigmaZ_g_10, seriestype=:scatter,color=:green,label="10°")
plot!(slices, Z_values_g_10,color=:green, label="kz = $kz_val_g_10")
plot!(slices, sigmaZ_g_12_5, seriestype=:scatter,color=:purple, label="12.5°")
plot!(slices, Z_values_g_12_5,color=:purple, label="kz = $kz_val_g_12_5", xlabel="X/D", ylabel="\\sigma_z", xrotation = 0,legend=:outertopright, legendfontsize=10, grid=false, dpi=300)
savefig("sigma_z_normal_g.png")

plot(slices, sigmaZ_2_5, seriestype=:scatter,color=:blue, label="2.5°")
plot!(slices, Z_values_2_5,color=:blue, label="kz = $kz_val_2_5")
plot!(slices, sigmaZ_5, seriestype=:scatter,color=:red, label="5°")
plot!(slices, Z_values_5,color=:red, label="kz = $kz_val_5")
plot!(slices, sigmaZ_7_5, seriestype=:scatter,color=:black, label="7.5°")
plot!(slices, Z_values_7_5,color=:black, label="kz = $kz_val_7_5")
plot!(slices, sigmaZ_10, seriestype=:scatter,color=:green,label="10°")
plot!(slices, Z_values_10,color=:green, label="kz = $kz_val_10")
plot!(slices, sigmaZ_12_5, seriestype=:scatter,color=:purple, label="12.5°")
plot!(slices, Z_values_12_5,color=:purple, label="kz = $kz_val_12_5")
plot!([0],[0],label=" ")
plot!(slices, sigmaZ_g_2_5, seriestype=:scatter,markershape=:star,color=:blue, label="2.5°")
plot!(slices, Z_values_g_2_5,color=:blue, label="kz = $kz_val_g_2_5")
plot!(slices, sigmaZ_g_5, seriestype=:scatter,markershape=:star,color=:red, label="5°")
plot!(slices, Z_values_g_5,color=:red, label="kz = $kz_val_g_5")
plot!(slices, sigmaZ_g_7_5, seriestype=:scatter,markershape=:star,color=:black, label="7.5°")
plot!(slices, Z_values_g_7_5,color=:black, label="kz = $kz_val_g_7_5")
plot!(slices, sigmaZ_g_10, seriestype=:scatter,markershape=:star,color=:green,label="10°")
plot!(slices, Z_values_g_10,color=:green, label="kz = $kz_val_g_10")
plot!(slices, sigmaZ_g_12_5, seriestype=:scatter,markershape=:star,color=:purple, label="12.5°")
plot!(slices, Z_values_g_12_5,color=:purple, label="kz = $kz_val_g_12_5", xlimits = [6, 11.5], ylimit = [0.18, 0.6],xlabel="X/D", ylabel="\\sigma_z/D", xrotation = 0,legend=:outertopright,legendtitle="Lower",legendtitlefontsize = 8, legendfontsize=7, grid=false, dpi=300)
savefig("sigma_z_both_crop.png")





"""SIGMA 0"""






# # Plot sigma_z0, sigma_y0
eps_z_up = [eps_z_2_5, eps_z_5, eps_z_7_5, eps_z_10, eps_z_12_5]
# eps_z_down = [eps_z_g_2_5, eps_z_g_5, eps_z_g_7_5, eps_z_g_10, eps_z_g_12_5]
eps_y = [eps_y_2_5, eps_y_5, eps_y_7_5, eps_y_10, eps_y_12_5]
TILT = [2.5, 5, 7.5, 10, 12.5]*(pi/180)

# tilt range 
tilt_plot = (2.5:0.1:12.5)*(pi/180)

# down
# Q = polyfit(TILT, eps_z_down, 2)
# sigz2 = Q[2]
# sigz1 = Q[1]
# eps_z = Q[0]
# Y_values_down = eps_z.+sigz1.*tilt_plot.+sigz2*(tilt_plot.^2)

# sigz2 = round(sigz2, digits=4)
# sigz1 = round(sigz1, digits=4)
# eps_z = round(eps_z, digits=4)
# up
# Q = polyfit(TILT, eps_z_up, 2)
# sigz2_up = Q[2]
# sigz1_up = Q[1]
# eps_z_u = Q[0]
# Y_values_up = eps_z_u.+sigz1_up.*tilt_plot.+sigz2_up*(tilt_plot.^2)

# sigz2_up = round(sigz2_up, digits=4)
# sigz1_up = round(sigz1_up, digits=4)
# eps_z_u = round(eps_z_u, digits=4)

# Q = curve_fit(LogFit, TILT.-(2.40*pi/180), eps_z_up)
sigz_up_a = 0.16848050034089787
# sigz_up_a = 0.16585649628045057
sigz_up_b = -0.013520451710140718
# sigz_up_b = -0.01478240211508666

Y_values_up = @. sigz_up_a + sigz_up_b*log(tilt_plot-(2.40*pi/180))
# y
Q = polyfit(TILT, eps_y, 2)
sigy2_d = Q[2]
sigy1_d = Q[1]
eps_y_d = Q[0]
Y_values = eps_y_d.+sigy1_d.*tilt_plot.+sigy2_d*(tilt_plot.^2)

sigy2_d = round(sigy2_d, digits=3)
sigy1_d = round(sigy1_d, digits=3)
eps_y_d = round(eps_y_d, digits=3)
sigz_up_plot = 0.013520451710140718
sigz_up_a = round(sigz_up_a, digits=3)
sigz_up_b = round(sigz_up_b, digits=3)
sigz_up_plot = round(sigz_up_plot, digits=3)

fontkz_c = text("", 12,color=:black).font
fontky_c = text("", 12,color=:dodgerblue).font
# plot(TILT, eps_y, seriestype=:scatter,color=:blue, label="\\sigma_y0")
# plot!(tilt_plot, Y_values, color=:blue, label="$eps_y_d + $sigy1_d\\gamma + $sigy2_d\\gamma^2")
# plot!(TILT, eps_z_up, seriestype=:scatter,color=:red, label="Upper \\sigma_{z0}")
# plot!(tilt_plot, Y_values_up, color=:red, label="$eps_z_u + $sigz1_up\\gamma + $sigz2_up\\gamma^2", xlabel="Tilt (radians)",legend=:outertopright, ylabel="\\sigma_0", xrotation = 0, legendfontsize=11, xticks=[0.0436, 0.0873, 0.1309, 0.1745, 0.2182], grid=false)
# # plot!(TILT, eps_z_down, seriestype=:scatter,color=:green, label="Lower \\sigma_{z0}")
# # plot!(tilt_plot, Y_values_down, color=:green, label="$eps_z + $sigz1\\gamma + $sigz2\\gamma^2", xlabel="Tilt (radians)",legend=:outertopright, ylabel="\\sigma_0", xrotation = 0, legendfontsize=11, xticks=[0.0436, 0.0873, 0.1309, 0.1745, 0.2182], grid=false, dpi=300)
# savefig("sigma_0.png")

plot(TILT*(180/pi), eps_y, seriestype=:scatter,color=:dodgerblue, label="\\sigma_y0")
plot!(tilt_plot*(180/pi), (tilt_plot./tilt_plot)*eps_y_d, color=:dodgerblue, label="$eps_y_d + $sigy1_d\\gamma + $sigy2_d\\gamma^2")
plot!(TILT*(180/pi), eps_z_up, seriestype=:scatter,color=:black, label="\\sigma_{z0}", yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz,foreground_color_legend = nothing)
plot!(tilt_plot*(180/pi), Y_values_up, color=:black,legend=:false, left_margin=7Plots.mm, xrotation = 0, xlabel="Tilt (degrees)", legendfontsize=11, xticks=[2.5, 5.0, 7.5, 10.0, 12.5], grid=false)
plot!(annotations = ([9.5], 0.21,text("\\sigma_{z0} = $sigz_up_a - $sigz_up_plot ln (\\gamma^{\\*})", fontkz_c)))
plot!(annotations = ([9.5], 0.25,text("\\sigma_{y0} = $eps_y_d", fontky_c)))
plot!(annotations = ([0.9], 0.23,text("\\sigma_0", fontylabel)))
# plot!(TILT*(180/pi), eps_z_down, seriestype=:scatter,color=:green, label="Lower \\sigma_{z0}")
# plot!(tilt_plot*(180/pi), Y_values_down, color=:green, label="$eps_z + $sigz1\\gamma + $sigz2\\gamma^2", xlabel="Tilt (degrees)",legend=:outertopright, ylabel="\\sigma_0", legendfontsize=11, xticks=[2.5, 5.0, 7.5, 10.0, 12.5], grid=false, dpi=300)
savefig("sigma_0_degree.pdf")






# """ky and kz as functions of tilt"""
Tilt = [2.5, 5, 7.5, 10, 12.5]*(pi/180)
Tilty = [2.5, 5, 7.5, 10, 12.5]*(pi/180)
tilt_plot = (2.5:0.1:12.5)*(pi/180)
Tilt_degree = [2.5, 5, 7.5, 10, 12.5]
ky_arr = [ky_val_2_5, ky_val_5, ky_val_7_5, ky_val_10, ky_val_12_5]
kz_arr = [kz_val_2_5, kz_val_5, kz_val_7_5, kz_val_10, kz_val_12_5]
# kz_arr_d = [kz_val_g_2_5, kz_val_g_5, kz_val_g_7_5, kz_val_g_10, kz_val_g_12_5]

Q = polyfit(Tilty, ky_arr, 1)
ky_val = Q[1]
eps_y = Q[0]
Z_values = eps_y.+ky_val.*tilt_plot

Q = polyfit(Tilt, kz_arr, 2)
kz1_val = Q[2]
kz_val = Q[1]
eps_z = Q[0]
Y_values = eps_z.+kz_val.*tilt_plot.+kz1_val.*(tilt_plot.^2)

# Q = polyfit(Tilt, kz_arr_d, 1)
# kz_val_d = Q[1]
# eps_z_d = Q[0]
# Y_values_d = eps_z_d.+kz_val_d.*Tilt

ky_val = round(ky_val, digits=3)
kz_val = round(kz_val, digits=3)
kz1_val = round(kz1_val, digits=3)
# kz_val_d = round(kz_val_d, digits=5)
eps_y = round(eps_y, digits=3)
eps_z = round(eps_z, digits=3)
fontkz_c = text("", 12,color=:black).font
fontky_c = text("", 12,color=:dodgerblue).font
# eps_z_d = round(eps_z_d, digits=5)
plot(Tilt_degree, ky_arr, seriestype=:scatter,color=:dodgerblue, label="ky = $ky_val\\gamma + $eps_y")
plot!(Tilt_degree, kz_arr, seriestype=:scatter,color=:black, label="kz = $kz1_val\\gamma^2 + $kz_val\\gamma + $eps_z", legend=:false, xlabel="Turbine Tilt (degrees)", grid=false, xticks=[2.5, 5.0, 7.5, 10.0, 12.5])
# plot!(Tilt_degree, kz_arr_d, seriestype=:scatter,color=:green, label="Lower kz")
plot!(tilt_plot.*(180/pi), Z_values, color=:dodgerblue, xtickfontsize=10, ytickfontsize=10, xguidefontsize=12, yguidefontsize=12)
plot!(tilt_plot.*(180/pi), Y_values, color=:black, legend=:false, left_margin=15Plots.mm, grid=:false, yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz,foreground_color_legend = nothing)
plot!(annotations = ([9.5], 0.0325,text("kz = $kz1_val\\gamma^{2} + $kz_val\\gamma + $eps_z", fontkz_c)))
plot!(annotations = ([8], 0.0225,text("ky = $ky_val\\gamma + $eps_y", fontky_c)))
plot!(annotations = ([-.1], 0.027,text("k-value", fontylabel)))
# plot!(Tilt_degree, Y_values_d,label="$kz_val_d\\gamma + $eps_z_d", color=:green, legend=:left, ylabel="k-value", xrotation = 0, xlabel="Turbine Tilt (degrees)", grid=false, xticks=[2.5, 5.0, 7.5, 10.0, 12.5], dpi=300)
savefig("kykztilt_both_new.pdf")

# plot(Tilt_degree*(pi/180), ky_arr, seriestype=:scatter,color=:blue, label="ky")
# plot!(Tilt_degree*(pi/180), kz_arr, seriestype=:scatter,color=:red, label="Upper kz")
# plot!(Tilt_degree*(pi/180), kz_arr_d, seriestype=:scatter,color=:green, label="Lower kz")
# plot!(Tilt_degree*(pi/180), Z_values,label="$ky_val\\gamma + $eps_y", color=:blue)
# plot!(Tilt_degree*(pi/180), Y_values,label="$kz_val\\gamma + $eps_z", color=:red)
# plot!(Tilt_degree*(pi/180), Y_values_d,label="$kz_val_d\\gamma + $eps_z_d", color=:green, legend=:left, ylabel="k-value", xrotation = 0, xlabel="Turbine Tilt (radians)", xticks=[0.0436, 0.0873, 0.1309, 0.1745, 0.2182], grid=false, dpi=300)
# savefig("kykztilt_both_new_radians.png")


"""DEFLECTION SURROGATE MODEL"""
# x_VAL = vcat(TILT, slices)
res1 = (defZ_2_5.-90)/D
res2 = (defZ_5.-90)/D
res3 = (defZ_7_5.-90)/D
res4 = (defZ_10.-90)/D
res5 = (defZ_12_5.-90)/D
Res = hcat(res1[6:end-15], res2[6:end-15], res3[6:end-15], res4[6:end-15], res5[6:end-15])
# Res = (defZ.-90)/D
# Res = Res[6:end-15]
guess0 = [0.2, -5.73, 0.1, -5.8, -9.4, -1.7]



slices_fine = 7.0:0.2:12.0
tilt = [2.5, 5, 7.5, 10, 12.5].*pi/180
tilt_fine = (2.5:0.1:12.5).*pi/180
Res_fine = interp2d(akima, slices_fine,tilt, Res, slices_fine, tilt_fine)
global Res_fine


"""Below is for finding the constants for the deflection surrogate model"""
Res_lin_solve = ones((length(Res_fine[:,1])*length(Res_fine[1,:])),1)
zz = 1
Trident = ones((length(Res_fine[:,1])*length(Res_fine[1,:])),8)
for i in 1:length(Res_fine[:,1])
    # i represents X/D
    # j represents Tilt
    for j in 1:length(Res_fine[1,:])
        Res_lin_solve[zz] = Res_fine[i,j]
        # Solve for each trident term
        # Tri1 = (tilt_fine[j])^2
        # Tri2 = tilt_fine[j]
        # Tri3 = (slices_fine[i])^2
        # Tri4 = slices_fine[i]
        # Tri5 = 1
        # Tri6 = exp(slices_fine[i])
        # Tri7 = exp(tilt_fine[j])
        # Tri8 = exp(-slices_fine[i])
        # Tri9 = exp(-tilt_fine[j])
        # Tri10 = slices_fine[i]*tilt_fine[j]
        # Trii = [Tri1, Tri2, Tri3, Tri4, Tri5, Tri6, Tri7, Tri8, Tri9, Tri10]
        # Trident[zz,:] = Trii

        Tri1 = (tilt_fine[j])^2
        Tri2 = tilt_fine[j]
        Tri3 = (slices_fine[i])^2
        Tri4 = slices_fine[i]
        Tri5 = 1
        Tri6 = ((tilt_fine[j])^2)*slices_fine[i]
        Tri7 = ((slices_fine[i])^2)*tilt_fine[j]
        Tri8 = (slices_fine[i]*tilt_fine[j])

        Trii = [Tri1, Tri2, Tri3, Tri4, Tri5, Tri6, Tri7, Tri8]
        Trident[zz,:] = Trii

        
        zz = zz + 1
    end
end

C = Trident\Res_lin_solve

function defz_2d(Tilt, X_D, C)
    def = C[1]*Tilt.^2 .+ C[2]*Tilt .+ C[3]*X_D.^2 .+ C[4]*X_D .+ C[5] .+ C[6].*(Tilt^2).*X_D .+ C[7].*(X_D.^2).*Tilt .+ C[8].*(X_D.*Tilt)
    return def
end

fit2d = zeros(size(Res_fine))
for i in 1:length(Res_fine[1,:])
    tt = tilt_fine[i]
    VAL = defz_2d(tt, slices_fine, C)
    print("VAL: ", VAL, "\n")
    fit2d[:,i] = VAL
end

fit4d = fit2d

font25_c = text("", 12, color=:lightskyblue2).font
font5_c = text("", 12,color=:skyblue2).font
font75_c = text("", 12,color=:steelblue2).font
font10_c = text("", 12,color=:dodgerblue).font
font125_c = text("", 12,color=:dodgerblue3).font


linew = 2.5
M = 15
X_ax = 12.4
plot([7,8,9,10,11,12], defz_2d(2.5*pi/180,[7,8,9,10,11,12], C), linewidth=linew,color=:lightskyblue2, label="2.5°")
plot!([7,8,9,10,11,12], defz_2d(5*pi/180, [7,8,9,10,11,12], C), linewidth=linew,color=:skyblue2, label="5°")
plot!([7,8,9,10,11,12], defz_2d(7.5*pi/180, [7,8,9,10,11,12], C), linewidth=linew,color=:steelblue2, label="7.5°")
plot!([7,8,9,10,11,12], defz_2d(10*pi/180, [7,8,9,10,11,12], C), linewidth=linew,color=:dodgerblue, label="10°")
plot!([7,8,9,10,11,12], defz_2d(12.5*pi/180, [7,8,9,10,11,12], C), linewidth=linew,color=:dodgerblue3, label="12.5°", legend=:false, xlimit=[7, 12.7], ylimit=[-0.36, -0.13], dpi=300)
plot!(slices[6:end-15], (defZ_2_5[6:end-15].-90)/D,seriestype=:scatter,color=:lightskyblue2, label="2.5°")
plot!(slices[6:end-15], (defZ_5[6:end-15].-90)/D,seriestype=:scatter,color=:skyblue2, label="5°")
plot!(slices[6:end-15], (defZ_7_5[6:end-15].-90)/D,seriestype=:scatter,color=:steelblue2, label="7.5°")
plot!(slices[6:end-15], (defZ_10[6:end-15].-90)/D,seriestype=:scatter,color=:dodgerblue, label="10°")
plot!(slices[6:end-15], (defZ_12_5[6:end-15].-90)/D,seriestype=:scatter,color=:dodgerblue3, label="12.5°", left_margin=7Plots.mm, xlabel="x/D", xrotation = 0,legend=:false,legendtitle="Surrogate", grid=false)
# plot!([7,8,9,10,11,12], defz_2d(-5, [7,8,9,10,11,12], RESULT))
# plot!([7,8,9,10,11,12], defz_2d(5, [7,8,9,10,11,12], RESULT))
# plot!([7,8,9,10,11,12], defz_2d(10, [7,8,9,10,11,12], RESULT))
# plot!([7,8,9,10,11,12], defz_2d(-5, [7,8,9,10,11,12], RESULT))
plot!([0], [0], label=" ")
plot!(annotations = ([6], -0.245,text("\\delta_z/D", fontylabel)))
plot!(annotations = ([X_ax], (defZ_2_5[end-M]-90)/D,text("2.5°", font25_c)))
plot!(annotations = ([X_ax], (defZ_5[end-M]-90)/D, text("5.0°", font5_c)))
plot!(annotations = ([X_ax], (defZ_7_5[end-M]-90)/D, text("7.5°", font75_c)))
plot!(annotations = ([X_ax], (defZ_10[end-M]-90)/D, text("10.0°", font10_c)))
plot!(annotations = ([X_ax], (defZ_12_5[end-M]-90)/D, text("12.5°", font125_c)), transparent=true, tickfontsize=12, guidefontsize=12, tickfont="Times New Roman")
savefig("Deflection_prediction_crop_surrogate.pdf")

# alpha = 1.0
# beta = 0.5
alpha = 2.32
beta = 0.154
I = 0.09
CT = 0.76
TILT = -[2.5, 5, 7.5, 10, 12.5]*(pi/180)
x_end = 12
ky = ky_arr
kz = kz_arr
sig0z = eps_z_up
sig0y = [eps_y_2_5, eps_y_5, eps_y_7_5, eps_y_10, eps_y_12_5]
# alpha = 2.32;
    # Beta = 0.154;
# tilt 2_5
defl_2_5, xvar_2_5 = deflection_call(1, ky, kz, sig0y, sig0z, TILT, I, CT, alpha, beta, x_end)

# tilt 5
defl_5, xvar_5 = deflection_call(2, ky, kz, sig0y, sig0z, TILT, I, CT, alpha, beta,x_end)

# tilt 7_5
defl_7_5, xvar_7_5 = deflection_call(3, ky, kz, sig0y, sig0z, TILT, I, CT, alpha, beta,x_end)

# tilt 10
defl_10, xvar_10 = deflection_call(4, ky, kz, sig0y, sig0z, TILT, I, CT, alpha, beta,x_end)

# tilt 12_5
defl_12_5, xvar_12_5 = deflection_call(5, ky, kz, sig0y, sig0z, TILT, I, CT, alpha, beta,x_end)



plot(slices[6:end-15], (defZ_2_5[6:end-15].-90)/D,seriestype=:scatter,color=:lightskyblue2, label="2.5°")
plot!(slices[6:end-15], (defZ_5[6:end-15].-90)/D,seriestype=:scatter,color=:skyblue2, label="5°")
plot!(slices[6:end-15], (defZ_7_5[6:end-15].-90)/D,seriestype=:scatter,color=:steelblue2, label="7.5°")
plot!(slices[6:end-15], (defZ_10[6:end-15].-90)/D,seriestype=:scatter,color=:dodgerblue, label="10°")
plot!(slices[6:end-15], (defZ_12_5[6:end-15].-90)/D,seriestype=:scatter,color=:dodgerblue3, label="12.5°", xlabel="x/D", xrotation = 0,legend=:outertopright,legendtitle="Surrogate", grid=false)


# plot!(xvar_2_5/D, defl_2_5,color=:lightskyblue2, label="2.5°")
# plot!(xvar_5/D, defl_5,color=:skyblue2, label="5°")
# plot!(xvar_7_5/D, defl_7_5,color=:steelblue2, label="7.5°")
# plot!(xvar_10/D, defl_10,color=:dodgerblue, label="10°")
# plot!(xvar_12_5/D, defl_12_5,color=:dodgerblue3, label="12.5°", dpi=300)

# plot!([7,8,9,10,11,12], defz_2d(-5, [7,8,9,10,11,12], RESULT))
# plot!([7,8,9,10,11,12], defz_2d(5, [7,8,9,10,11,12], RESULT))
# plot!([7,8,9,10,11,12], defz_2d(10, [7,8,9,10,11,12], RESULT))
# plot!([7,8,9,10,11,12], defz_2d(-5, [7,8,9,10,11,12], RESULT))
linew = 2.5
plot!([0], [0], label=" ")
plot!(xvar_2_5/D, defl_2_5,color=:lightskyblue2, linewidth=linew, label="2.5°")
plot!(xvar_5/D, defl_5,color=:skyblue2,linewidth=linew, label="5°")
plot!(xvar_7_5/D, defl_7_5,color=:steelblue2,linewidth=linew, label="7.5°")
plot!(xvar_10/D, defl_10,color=:dodgerblue,linewidth=linew, label="10°")
plot!(xvar_12_5/D, defl_12_5,color=:dodgerblue3,linewidth=linew, label="12.5°",left_margin=7Plots.mm, legend=:false, xlimit=[7, 12.7], ylimit=[-0.37, 0.0], dpi=300)
# plot!(annotations = ([X_ax], (defZ_2_5[end-M]-90)/D,text("2.5°", font25_c)))
# plot!(annotations = ([X_ax], (defZ_5[end-M]-90)/D, text("5.0°", font5_c)))
# plot!(annotations = ([X_ax], (defZ_7_5[end-M]-90)/D, text("7.5°", font75_c)))
# plot!(annotations = ([X_ax], (defZ_10[end-M]-90)/D, text("10.0°", font10_c)))
# plot!(annotations = ([X_ax], (defZ_12_5[end-M]-90)/D, text("12.5°", font125_c)), transparent=true, tickfontsize=12, guidefontsize=12)
plot!(annotations = ([X_ax], defl_2_5[end],text("2.5°", font25_c)))
plot!(annotations = ([6.14], -0.19,text("\\delta_z/D", fontylabel)))
plot!(annotations = ([X_ax], defl_5[end], text("5.0°", font5_c)))
plot!(annotations = ([X_ax], defl_7_5[end], text("7.5°", font75_c)))
plot!(annotations = ([X_ax], defl_10[end], text("10.0°", font10_c)))
plot!(annotations = ([X_ax], defl_12_5[end], text("12.5°", font125_c)), transparent=true, tickfontsize=12, guidefontsize=12)
savefig("Deflection_prediction_crop.pdf")
# """Max deflection vs. Tilt"""
# Tilt = [2.5, 5.0, 7.5, 10.0, 12.5]
# Max_deflection = [(defZ_2_5[end-18]-90)/D, (defZ_5[end-18]-90)/D, (defZ_7_5[end-18]-90)/D, (defZ_10[end-18]-90)/D, (defZ_12_5[end-18]-90)/D]

# plot(Tilt, Max_deflection, seriestype=:scatter, xticks=[2.5, 5.0, 7.5, 10.0, 12.5], color=:steelblue2, legend=:false, grid=false, xlabel="Tilt Angle (degrees)", ylabel="\\delta_z/{D}", xrotation = 0,size=((10/8)*400,400),yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz,foreground_color_legend = nothing)





























# fit = curve_fit(fit_model, x_VAL, Res, guess0)


# # Sensitivity analysis
# smoothstep = 200
# deficitstep = 0.1
# # Derivative of 0.5 m/s threshold at 7 D downstream
# # Z deflection w.r.t smooth
# defz_1 = defZ[1,:,1]
# defz_n = defz_1[2:end]
# defz_p = defz_1[1:end-1]

# # Z deflection w.r.t. threshold at 7 D downstream
# defz_1_t = defZ[1,1,:]
# defz_1_tn = defz_1_t[2:end]
# defz_1_tp = defz_1_t[1:end-1]

# # Native Sensitivity
# dzds_1 = (defz_n-defz_p)/200
# dzdt_1 = (defz_1_tn-defz_1_tp)/deficitstep

# # Normalized Sensitivity
# dzds_norm_1 = (smoothing[1]/defz_1[1])*dzds_1
# dzdt_norm_1 = (threshold[1]/defz_1_t[1])*dzdt_1

# # Derivative of 0.5 m/s threshold at 10 D downstream
# # Z deflection w.r.t smooth
# defz_1 = defZ[2,:,1]
# defz_n = defz_1[2:end]
# defz_p = defz_1[1:end-1]

# # Z deflection w.r.t. threshold at 10 D downstream
# defz_1_t = defZ[2,1,:]
# defz_1_tn = defz_1_t[2:end]
# defz_1_tp = defz_1_t[1:end-1]

# # Native Sensitivity
# dzds_1 = (defz_n-defz_p)/200
# dzdt_1 = (defz_1_tn-defz_1_tp)/deficitstep

# # Normalized Sensitivity
# dzds_norm_2 = (smoothing[1]/defz_1[1])*dzds_1
# dzdt_norm_2 = (threshold[1]/defz_1_t[1])*dzdt_1

# # Derivative of 0.5 m/s threshold at 13 D downstream
# # Z deflection w.r.t smooth
# defz_1 = defZ[3,:,1]
# defz_n = defz_1[2:end]
# defz_p = defz_1[1:end-1]

# # Z deflection w.r.t. threshold at 13 D downstream
# defz_1_t = defZ[3,1,:]
# defz_1_tn = defz_1_t[2:end]
# defz_1_tp = defz_1_t[1:end-1]

# # Native Sensitivity
# dzds_1 = (defz_n-defz_p)/200
# dzdt_1 = (defz_1_tn-defz_1_tp)/deficitstep

# # Normalized Sensitivity
# dzds_norm_3 = (smoothing[1]/defz_1[1])*dzds_1
# dzdt_norm_3 = (threshold[1]/defz_1_t[1])*dzdt_1

# # Derivative of 0.5 m/s threshold at 15 D downstream
# # Z deflection w.r.t smooth
# defz_1 = defZ[4,:,1]
# defz_n = defz_1[2:end]
# defz_p = defz_1[1:end-1]

# # Z deflection w.r.t. threshold at 15 D downstream
# defz_1_t = defZ[4,1,:]
# defz_1_tn = defz_1_t[2:end]
# defz_1_tp = defz_1_t[1:end-1]

# # Native Sensitivity
# dzds_1 = (defz_n-defz_p)/200
# dzdt_1 = (defz_1_tn-defz_1_tp)/deficitstep

# # Normalized Sensitivity
# dzds_norm_4 = (smoothing[1]/defz_1[1])*dzds_1
# dzdt_norm_4 = (threshold[1]/defz_1_t[1])*dzdt_1

# # heatmap
# thresh = LinRange(minimum(threshold),maximum(threshold),length(threshold))
# smooth = LinRange(minimum(smoothing),maximum(smoothing),length(smoothing))
# DEFZ = hcat(defZ[1,:,:], zeros(51,42))
# THRESH = vcat(thresh, zeros(42))
# data1 = heatmap(thresh, smooth, defZ[1,:,:])

# # make square for heatmap

# # boxplot(["7D"],dzds_norm_1)
# # boxplot!(["10D"],dzds_norm_2)
# # boxplot!(["13D"],dzds_norm_3)
# # boxplot!(["15D"],dzds_norm_4, xlabel = "Downstream Distance", ylabel="\\ s_0/Z_0*dZ/ds)", xrotation = 0,legend=:false)
# # savefig("dzds.png")

# # boxplot(["7D"],dzdt_norm_1)
# # boxplot!(["10D"],dzdt_norm_2)
# # boxplot!(["13D"],dzdt_norm_3)
# # boxplot!(["15D"],dzdt_norm_4,ylimit=[-0.15,0.025], xlabel = "Downstream Distance", ylabel="\\ t_0/Z_0*dZ/dt", xrotation = 0,legend=:false)
# # savefig("dzdt.png")

# # boxplot(["7D"],dzds_norm_1)
# # boxplot!(["10D"],dzds_norm_2)
# # boxplot!(["13D"],dzds_norm_3)
# # boxplot!(["15D"],dzds_norm_4, xlabel = "Downstream Distance", ylabel="\\ s_0/Y_0*dy/Ds", xrotation = 0,legend=:false)
# # savefig("dyds.png")

# # boxplot(["7D"],dzdt_norm_1)
# # boxplot!(["10D"],dzdt_norm_2)
# # boxplot!(["13D"],dzdt_norm_3)
# # boxplot!(["15D"],dzdt_norm_4, xlabel = "Downstream Distance", ylabel="\\ t_0/Y_0*dy/Dt", xrotation = 0,legend=:false)
# # savefig("dydt.png")

# # boxplot(["7D"],dzds_norm_1)
# # boxplot!(["10D"],dzds_norm_2)
# # boxplot!(["13D"],dzds_norm_3)
# # boxplot!(["15D"],dzds_norm_4, xlabel = "Downstream Distance", ylabel="(s_0/\\sigma_z0*d\\sigma_z/ds)", xrotation = 0,legend=:false)
# # savefig("d\\sigma_zds.png")

# # boxplot(["7D"],dzdt_norm_1)
# # boxplot!(["10D"],dzdt_norm_2)
# # boxplot!(["13D"],dzdt_norm_3)
# # boxplot!(["15D"],dzdt_norm_4, xlabel = "Downstream Distance",ylimit=[-0.05,0.1], ylabel="(t_0/\\sigma_z0*d\\sigma_z/dt)", xrotation = 0,legend=:false)
# # savefig("d\\sigma_zdt.png")

# boxplot(["7D"],dzds_norm_1)
# boxplot!(["10D"],dzds_norm_2)
# boxplot!(["13D"],dzds_norm_3)
# boxplot!(["15D"],dzds_norm_4, xlabel = "Downstream Distance", ylabel="(s_0/\\sigma_y0*d\\sigma_y/Ds)", xrotation = 0,legend=:false)
# savefig("d\\sigma_yds.png")

# boxplot(["7D"],dzdt_norm_1)
# boxplot!(["10D"],dzdt_norm_2)
# boxplot!(["13D"],dzdt_norm_3)
# boxplot!(["15D"],dzdt_norm_4, xlabel = "Downstream Distance", ylabel="(t_0/\\sigma_y0*d\\sigma_y/Dt)", xrotation = 0,legend=:false)
# savefig("d\\sigma_ydt.png")
# # Y deflection
# heatmap((defY[4,:,:].-2500)/D, right_margin=5Plots.mm)

# # sigy

# # sigz

# # Derivative of 0.6 m/s threshold at 7 D downstream

# # Derivative of 0.7 m/s threshold at 7 D downstream

# # Derivative of 0.8 m/s threshold at 7 D downstream

# # Derivative of 0.9 m/s threshold at 7 D downstream

# # Derivative of 1.0 m/s threshold at 7 D downstream

# # Derivative of 1.1 m/s threshold at 7 D downstream

# # Derivative of 1.2 m/s threshold at 7 D downstream

# # Derivative of 1.3 m/s threshold at 7 D downstream

# # FACTORS IN ANALYSIS
# # Threshold Velocity Deficit
# # Moving Averaging Window


# # t1 = threshold[1]
# # t2 = threshold[2]
# # t3 = threshold[3]
# # t4 = threshold[4]
# # t5 = threshold[5]
# # t6 = threshold[6]
# # t7 = threshold[7]
# # t8 = threshold[8]
# # t9 = threshold[9]
# # t10 = threshold[10]
# # t11 = threshold[11]
# # t12 = threshold[12]

# # # deflection Z hybrid gauss threshold
# # defz = defZ[:,5,:]
# # plot(slice, defz[:,1]/90, color=:lightsalmon, label="$t1 m/s",markershape=:square)
# # plot!(slice, defz[:,2]/90, color=:orange, label="$t2 m/s",markershape=:square)
# # plot!(slice, defz[:,3]/90, color=:lightseagreen, label="$t3 m/s",markershape=:square)
# # plot!(slice, defz[:,4]/90, color=:green, label="$t4 m/s",markershape=:square)
# # plot!(slice, defz[:,5]/90, color=:dodgerblue, label="$t5 m/s",markershape=:square)
# # plot!(slice, defz[:,6]/90, color=:darkmagenta, label="$t6 m/s",markershape=:square)
# # plot!(slice, defz[:,7]/90, color=:blue, label="$t7 m/s",markershape=:square)
# # plot!(slice, defz[:,8]/90, color=:chocolate1, label="$t8 m/s",markershape=:square)
# # plot!(slice, defz[:,9]/90, color=:grey0, label="$t9 m/s",markershape=:square, xlabel="X/D", ylabel="z*", xrotation = 0, legendtitle="Threshold", legend=:outertopright)


# # # Smoothing effect on deflection_Z
# # plot(smoothing, defZ[1,:,9]/90, color=:lightsalmon, label="5D",markershape=:square)
# # plot!(smoothing, defZ[2,:,9]/90, color=:orange, label="6D",markershape=:square)
# # plot!(smoothing, defZ[3,:,9]/90, color=:lightseagreen, label="7D",markershape=:square)
# # plot!(smoothing, defZ[4,:,9]/90, color=:green, label="8D",markershape=:square)
# # plot!(smoothing, defZ[5,:,9]/90, color=:dodgerblue, label="9D",markershape=:square)
# # plot!(smoothing, defZ[6,:,9]/90, color=:darkmagenta, label="10D",markershape=:square)
# # plot!(smoothing, defZ[7,:,9]/90, color=:blue, label="11D",markershape=:square)
# # plot!(smoothing, defZ[8,:,9]/90, color=:chocolate1, label="12D",markershape=:square)
# # plot!(smoothing, defZ[9,:,9]/90, color=:firebrick, label="13D",markershape=:square)
# # plot!(smoothing, defZ[10,:,9]/90, color=:olive, label="11D",markershape=:square)
# # plot!(smoothing, defZ[11,:,9]/90, color=:orange, label="15D",markershape=:square, xlabel="Averaging Window Size", ylabel="z*", xrotation = 0, legendtitle="Downstream Distance", legend=:outertopright)
# # savefig("smoothvsdefz_9.png")

# # # Smoothing effect on deflection_Y
# # plot(smoothing, (defY[1,:,1].-2500)/D, color=:lightsalmon, label="5D",markershape=:square)
# # plot!(smoothing, (defY[2,:,1].-2500)/D, color=:orange, label="6D",markershape=:square)
# # plot!(smoothing, (defY[3,:,1].-2500)/D, color=:lightseagreen, label="7D",markershape=:square)
# # plot!(smoothing, (defY[4,:,1].-2500)/D, color=:green, label="8D",markershape=:square)
# # plot!(smoothing, (defY[5,:,1].-2500)/D, color=:dodgerblue, label="9D",markershape=:square)
# # plot!(smoothing, (defY[6,:,1].-2500)/D, color=:darkmagenta, label="10D",markershape=:square)
# # plot!(smoothing, (defY[7,:,1].-2500)/D, color=:blue, label="11D",markershape=:square)
# # plot!(smoothing, (defY[8,:,1].-2500)/D, color=:chocolate1, label="12D",markershape=:square)
# # plot!(smoothing, (defY[9,:,1].-2500)/D, color=:firebrick, label="13D",markershape=:square)
# # plot!(smoothing, (defY[10,:,1].-2500)/D, color=:olive, label="11D",markershape=:square)
# # plot!(smoothing, (defY[11,:,1].-2500)/D, color=:orange, label="15D",markershape=:square, xlabel="Averaging Window Size", ylabel="y/D", xrotation = 0, legendtitle="Downstream Distance", legend=:outertopright)
# # savefig("smoothvsdefy_1.png")

# # # Smoothing effect on deflection_Y
# # plot(smoothing, sigmaY[1,:,1], color=:lightsalmon, label="5D",markershape=:square)
# # plot!(smoothing, sigmaY[2,:,1], color=:orange, label="6D",markershape=:square)
# # plot!(smoothing, sigmaY[3,:,1], color=:lightseagreen, label="7D",markershape=:square)
# # plot!(smoothing, sigmaY[4,:,1], color=:green, label="8D",markershape=:square)
# # plot!(smoothing, sigmaY[5,:,1], color=:dodgerblue, label="9D",markershape=:square)
# # plot!(smoothing, sigmaY[6,:,1], color=:darkmagenta, label="10D",markershape=:square)
# # plot!(smoothing, sigmaY[7,:,1], color=:blue, label="11D",markershape=:square)
# # plot!(smoothing, sigmaY[8,:,1], color=:chocolate1, label="12D",markershape=:square)
# # plot!(smoothing, sigmaY[9,:,1], color=:firebrick, label="13D",markershape=:square)
# # plot!(smoothing, sigmaY[10,:,1], color=:olive, label="11D",markershape=:square)
# # plot!(smoothing, sigmaY[11,:,1], color=:orange, label="15D",markershape=:square, xlabel="Averaging Window Size", ylabel="\\sigma_y/D", xrotation = 0, legendtitle="Downstream Distance", legend=:outertopright)
# # savefig("smoothvssigy.png")

# # # Smoothing effect on sigmaz
# # plot(smoothing, sigmaZ[1,:,9], color=:lightsalmon, label="5D",markershape=:square)
# # plot!(smoothing, sigmaZ[2,:,9], color=:orange, label="6D",markershape=:square)
# # plot!(smoothing, sigmaZ[3,:,9], color=:lightseagreen, label="7D",markershape=:square)
# # plot!(smoothing, sigmaZ[4,:,9], color=:green, label="8D",markershape=:square)
# # plot!(smoothing, sigmaZ[5,:,9], color=:dodgerblue, label="9D",markershape=:square)
# # plot!(smoothing, sigmaZ[6,:,9], color=:darkmagenta, label="10D",markershape=:square)
# # plot!(smoothing, sigmaZ[7,:,9], color=:blue, label="11D",markershape=:square)
# # plot!(smoothing, sigmaZ[8,:,9], color=:chocolate1, label="12D",markershape=:square)
# # plot!(smoothing, sigmaZ[9,:,9], color=:firebrick, label="13D",markershape=:square)
# # plot!(smoothing, sigmaZ[10,:,9], color=:olive, label="11D",markershape=:square)
# # plot!(smoothing, sigmaZ[11,:,9], color=:orange, label="15D",markershape=:square, xlabel="Averaging Window Size", ylabel="\\sigma_z/D", xrotation = 0, legendtitle="Downstream Distance", legend=:outertopright)
# # savefig("smoothvssigz_9.png")

# # # Threshold effect on deflection_Z
# # plot(threshold, defZ[1,6,:]/90, color=:lightsalmon, label="5D",markershape=:square)
# # plot!(threshold, defZ[2,6,:]/90, color=:orange, label="6D",markershape=:square)
# # plot!(threshold, defZ[3,6,:]/90, color=:lightseagreen, label="7D",markershape=:square)
# # plot!(threshold, defZ[4,6,:]/90, color=:green, label="8D",markershape=:square)
# # plot!(threshold, defZ[5,6,:]/90, color=:dodgerblue, label="9D",markershape=:square)
# # plot!(threshold, defZ[6,6,:]/90, color=:darkmagenta, label="10D",markershape=:square)
# # plot!(threshold, defZ[7,6,:]/90, color=:blue, label="11D",markershape=:square)
# # plot!(threshold, defZ[8,6,:]/90, color=:chocolate1, label="12D",markershape=:square)
# # plot!(threshold, defZ[9,6,:]/90, color=:firebrick, label="13D",markershape=:square)
# # plot!(threshold, defZ[10,6,:]/90, color=:olive, label="11D",markershape=:square)
# # plot!(threshold, defZ[11,6,:]/90, color=:orange, label="15D",markershape=:square, xlabel="Threshold Value (m/s)", ylabel="z*", xrotation = 0, legendtitle="Downstream Distance", legend=:outertopright)
# # savefig("Threshvsdefz_6.png")

# # # Threshold effect on deflection_Y
# # plot(threshold, (defY[1,1,:].-2500)/D, color=:lightsalmon, label="5D",markershape=:square)
# # plot!(threshold, (defY[2,1,:].-2500)/D, color=:orange, label="6D",markershape=:square)
# # plot!(threshold, (defY[3,1,:].-2500)/D, color=:lightseagreen, label="7D",markershape=:square)
# # plot!(threshold, (defY[4,1,:].-2500)/D, color=:green, label="8D",markershape=:square)
# # plot!(threshold, (defY[5,1,:].-2500)/D, color=:dodgerblue, label="9D",markershape=:square)
# # plot!(threshold, (defY[6,1,:].-2500)/D, color=:darkmagenta, label="10D",markershape=:square)
# # plot!(threshold, (defY[7,1,:].-2500)/D, color=:blue, label="11D",markershape=:square)
# # plot!(threshold, (defY[8,1,:].-2500)/D, color=:chocolate1, label="12D",markershape=:square)
# # plot!(threshold, (defY[9,1,:].-2500)/D, color=:firebrick, label="13D",markershape=:square)
# # plot!(threshold, (defY[10,1,:].-2500)/D, color=:olive, label="11D",markershape=:square)
# # plot!(threshold, (defY[11,1,:].-2500)/D, color=:orange, label="15D",markershape=:square, xlabel="Threshold Value (m/s)", ylabel="y/D", xrotation = 0, legendtitle="Downstream Distance", legend=:outertopright)
# # savefig("Threshvsdefy_1.png")

# # # Threshold effect on sigma_z
# # plot(threshold, sigmaZ[1,1,:], color=:lightsalmon, label="5D",markershape=:square)
# # plot!(threshold, sigmaZ[2,1,:], color=:orange, label="6D",markershape=:square)
# # plot!(threshold, sigmaZ[3,1,:], color=:lightseagreen, label="7D",markershape=:square)
# # plot!(threshold, sigmaZ[4,1,:], color=:green, label="8D",markershape=:square)
# # plot!(threshold, sigmaZ[5,1,:], color=:dodgerblue, label="9D",markershape=:square)
# # plot!(threshold, sigmaZ[6,1,:], color=:darkmagenta, label="10D",markershape=:square)
# # plot!(threshold, sigmaZ[7,1,:], color=:blue, label="11D",markershape=:square)
# # plot!(threshold, sigmaZ[8,1,:], color=:chocolate1, label="12D",markershape=:square)
# # plot!(threshold, sigmaZ[9,1,:], color=:firebrick, label="13D",markershape=:square)
# # plot!(threshold, sigmaZ[10,1,:], color=:olive, label="11D",markershape=:square)
# # plot!(threshold, sigmaZ[11,1,:], color=:orange, label="15D",markershape=:square, xlabel="Threshold Value (m/s)", ylabel="\\sigma_z/D", xrotation = 0, legendtitle="Downstream Distance", legend=:outertopright)
# # savefig("Threshvssigmaz_1.png")

# # # Threshold effect on sigma_y
# # plot(threshold, sigmaY[1,6,:], color=:lightsalmon, label="5D",markershape=:square)
# # plot!(threshold, sigmaY[2,6,:], color=:orange, label="6D",markershape=:square)
# # plot!(threshold, sigmaY[3,6,:], color=:lightseagreen, label="7D",markershape=:square)
# # plot!(threshold, sigmaY[4,6,:], color=:green, label="8D",markershape=:square)
# # plot!(threshold, sigmaY[5,6,:], color=:dodgerblue, label="9D",markershape=:square)
# # plot!(threshold, sigmaY[6,6,:], color=:darkmagenta, label="10D",markershape=:square)
# # plot!(threshold, sigmaY[7,6,:], color=:blue, label="11D",markershape=:square)
# # plot!(threshold, sigmaY[8,6,:], color=:chocolate1, label="12D",markershape=:square)
# # plot!(threshold, sigmaY[9,6,:], color=:firebrick, label="13D",markershape=:square)
# # plot!(threshold, sigmaY[10,6,:], color=:olive, label="11D",markershape=:square)
# # plot!(threshold, sigmaY[11,6,:], color=:orange, label="15D",markershape=:square, xlabel="Threshold Value (m/s)", ylabel="\\sigma_y/D", xrotation = 0, legendtitle="Downstream Distance", legend=:outertopright)
# # savefig("Threshvssigmay_6.png")
# # plot!(slice, defz[:,10]/90, color=:firebrick, label="$t10 m/s",markershape=:square)
# # plot!(slice, defz[:,11]/90, color=:orange3, label="$t11 m/s",markershape=:square)
# # plot!(slice, defz[:,12]/90, color=:olive, markershape=:square, label="$t12 m/s", xlabel="X/D", ylabel="z*", xrotation = 0, legendtitle="Threshold", legend=:outertopright)

# # defz = defZ[:,2,:]
# # plot!(slice, defz[:,1]/90, color=:lightsalmon, label="$t1 m/s",markershape=:square)
# # plot!(slice, defz[:,2]/90, color=:orange, label="$t2 m/s",markershape=:square)
# # plot!(slice, defz[:,3]/90, color=:lightseagreen, label="$t3 m/s",markershape=:square)
# # plot!(slice, defz[:,4]/90, color=:green, label="$t4 m/s",markershape=:square)
# # plot!(slice, defz[:,5]/90, color=:dodgerblue, label="$t5 m/s",markershape=:square)
# # plot!(slice, defz[:,6]/90, color=:darkmagenta, label="$t6 m/s",markershape=:square)
# # plot!(slice, defz[:,7]/90, color=:maroon, label="$t7 m/s",markershape=:square)
# # plot!(slice, defz[:,8]/90, color=:chocolate1, label="$t8 m/s",markershape=:square)
# # plot!(slice, defz[:,9]/90, color=:grey0, label="$t9 m/s",markershape=:square)
# # plot!(slice, defz[:,10]/90, color=:firebrick, label="$t10 m/s",markershape=:square)
# # plot!(slice, defz[:,11]/90, color=:orange3, label="$t11 m/s",markershape=:square)
# # plot!(slice, defz[:,12]/90, color=:olive, markershape=:square, label="$t12 m/s", xlabel="X/D", ylabel="z*", xrotation = 0, legendtitle="Threshold", legend=:outertopright)

# # defz = defZ[:,3,:]
# # plot!(slice, defz[:,1]/90, color=:lightsalmon, label="$t1 m/s",markershape=:square)
# # plot!(slice, defz[:,2]/90, color=:orange, label="$t2 m/s",markershape=:square)
# # plot!(slice, defz[:,3]/90, color=:lightseagreen, label="$t3 m/s",markershape=:square)
# # plot!(slice, defz[:,4]/90, color=:green, label="$t4 m/s",markershape=:square)
# # plot!(slice, defz[:,5]/90, color=:dodgerblue, label="$t5 m/s",markershape=:square)
# # plot!(slice, defz[:,6]/90, color=:darkmagenta, label="$t6 m/s",markershape=:square)
# # plot!(slice, defz[:,7]/90, color=:maroon, label="$t7 m/s",markershape=:square)
# # plot!(slice, defz[:,8]/90, color=:chocolate1, label="$t8 m/s",markershape=:square)
# # plot!(slice, defz[:,9]/90, color=:grey0, label="$t9 m/s",markershape=:square)
# # plot!(slice, defz[:,10]/90, color=:firebrick, label="$t10 m/s",markershape=:square)
# # plot!(slice, defz[:,11]/90, color=:orange3, label="$t11 m/s",markershape=:square)
# # plot!(slice, defz[:,12]/90, color=:olive, markershape=:square, label="$t12 m/s", xlabel="X/D", ylabel="z*", xrotation = 0, legendtitle="Threshold", legend=:outertopright)

# # defz = defZ[:,4,:]
# # plot(slice, defz[:,1]/90, color=:lightsalmon, label="$t1 m/s",markershape=:square)
# # plot!(slice, defz[:,2]/90, color=:orange, label="$t2 m/s",markershape=:square)
# # plot!(slice, defz[:,3]/90, color=:lightseagreen, label="$t3 m/s",markershape=:square)
# # plot!(slice, defz[:,4]/90, color=:green, label="$t4 m/s",markershape=:square)
# # plot!(slice, defz[:,5]/90, color=:dodgerblue, label="$t5 m/s",markershape=:square)
# # plot!(slice, defz[:,6]/90, color=:darkmagenta, label="$t6 m/s",markershape=:square)
# # plot!(slice, defz[:,7]/90, color=:maroon, label="$t7 m/s",markershape=:square)
# # plot!(slice, defz[:,8]/90, color=:chocolate1, label="$t8 m/s",markershape=:square)
# # plot!(slice, defz[:,9]/90, color=:grey0, label="$t9 m/s",markershape=:square)
# # plot!(slice, defz[:,10]/90, color=:firebrick, label="$t10 m/s",markershape=:square)
# # plot!(slice, defz[:,11]/90, color=:orange3, label="$t11 m/s",markershape=:square)
# # plot!(slice, defz[:,12]/90, color=:olive, markershape=:square, label="$t12 m/s", xlabel="X/D", ylabel="z*", xrotation = 0, legendtitle="Threshold", legend=:outertopright)

# # defz = defZ[:,5,:]
# # plot(slice, defz[:,1]/90, color=:lightsalmon, label="$t1 m/s",markershape=:square)
# # plot!(slice, defz[:,2]/90, color=:orange, label="$t2 m/s",markershape=:square)
# # plot!(slice, defz[:,3]/90, color=:lightseagreen, label="$t3 m/s",markershape=:square)
# # plot!(slice, defz[:,4]/90, color=:green, label="$t4 m/s",markershape=:square)
# # plot!(slice, defz[:,5]/90, color=:dodgerblue, label="$t5 m/s",markershape=:square)
# # plot!(slice, defz[:,6]/90, color=:darkmagenta, label="$t6 m/s",markershape=:square)
# # plot!(slice, defz[:,7]/90, color=:maroon, label="$t7 m/s",markershape=:square)
# # plot!(slice, defz[:,8]/90, color=:chocolate1, label="$t8 m/s",markershape=:square)
# # plot!(slice, defz[:,9]/90, color=:grey0, label="$t9 m/s",markershape=:square)
# # plot!(slice, defz[:,10]/90, color=:firebrick, label="$t10 m/s",markershape=:square)
# # plot!(slice, defz[:,11]/90, color=:orange3, label="$t11 m/s",markershape=:square)
# # plot!(slice, defz[:,12]/90, color=:olive, markershape=:square, label="$t12 m/s", xlabel="X/D", ylabel="z*", xrotation = 0, legendtitle="Threshold", legend=:outertopright)

# # defavg = (defz[:,1].+defz[:,2].+defz[:,3].+defz[:,4].+defz[:,5].+defz[:,6].+defz[:,7].+defz[:,8].+defz[:,9].+defz[:,10].+defz[:,11].+defz[:,12])/12
# # defz = defZ[:,6,:]
# # plot!(slice, defz[:,1]/90, color=:lightsalmon, label="$t1 m/s",markershape=:square)
# # plot!(slice, defz[:,2]/90, color=:orange, label="$t2 m/s",markershape=:square)
# # plot!(slice, defz[:,3]/90, color=:lightseagreen, label="$t3 m/s",markershape=:square)
# # plot!(slice, defz[:,4]/90, color=:green, label="$t4 m/s",markershape=:square)
# # plot!(slice, defz[:,5]/90, color=:dodgerblue, label="$t5 m/s",markershape=:square)
# # plot!(slice, defz[:,6]/90, color=:darkmagenta, label="$t6 m/s",markershape=:square)
# # plot!(slice, defz[:,7]/90, color=:maroon, label="$t7 m/s",markershape=:square)
# # plot!(slice, defz[:,8]/90, color=:chocolate1, label="$t8 m/s",markershape=:square)
# # plot!(slice, defz[:,9]/90, color=:grey0, label="$t9 m/s",markershape=:square)
# # plot!(slice, defz[:,10]/90, color=:firebrick, label="$t10 m/s",markershape=:square)
# # plot!(slice, defz[:,11]/90, color=:orange3, label="$t11 m/s",markershape=:square)
# # plot!(slice, defz[:,12]/90, color=:olive, markershape=:square, label="$t12 m/s", xlabel="X/D", ylabel="z*", xrotation = 0, legendtitle="Threshold", legend=:outertopright)

# # defz = defZ[:,7,:]
# # plot!(slice, defz[:,1]/90, color=:lightsalmon, label="$t1 m/s",markershape=:square)
# # plot!(slice, defz[:,2]/90, color=:orange, label="$t2 m/s",markershape=:square)
# # plot!(slice, defz[:,3]/90, color=:lightseagreen, label="$t3 m/s",markershape=:square)
# # plot!(slice, defz[:,4]/90, color=:green, label="$t4 m/s",markershape=:square)
# # plot!(slice, defz[:,5]/90, color=:dodgerblue, label="$t5 m/s",markershape=:square)
# # plot!(slice, defz[:,6]/90, color=:darkmagenta, label="$t6 m/s",markershape=:square)
# # plot!(slice, defz[:,7]/90, color=:maroon, label="$t7 m/s",markershape=:square)
# # plot!(slice, defz[:,8]/90, color=:chocolate1, label="$t8 m/s",markershape=:square)
# # plot!(slice, defz[:,9]/90, color=:grey0, label="$t9 m/s",markershape=:square)
# # plot!(slice, defz[:,10]/90, color=:firebrick, label="$t10 m/s",markershape=:square)
# # plot!(slice, defz[:,11]/90, color=:orange3, label="$t11 m/s",markershape=:square)
# # plot!(slice, defz[:,12]/90, color=:olive, markershape=:square, label="$t12 m/s", xlabel="X/D", ylabel="z*", xrotation = 0, legendtitle="Threshold", legend=:outertopright)

# # defz = defZ[:,8,:]
# # plot!(slice, defz[:,1]/90, color=:lightsalmon, label="$t1 m/s",markershape=:square)
# # plot!(slice, defz[:,2]/90, color=:orange, label="$t2 m/s",markershape=:square)
# # plot!(slice, defz[:,3]/90, color=:lightseagreen, label="$t3 m/s",markershape=:square)
# # plot!(slice, defz[:,4]/90, color=:green, label="$t4 m/s",markershape=:square)
# # plot!(slice, defz[:,5]/90, color=:dodgerblue, label="$t5 m/s",markershape=:square)
# # plot!(slice, defz[:,6]/90, color=:darkmagenta, label="$t6 m/s",markershape=:square)
# # plot!(slice, defz[:,7]/90, color=:maroon, label="$t7 m/s",markershape=:square)
# # plot!(slice, defz[:,8]/90, color=:chocolate1, label="$t8 m/s",markershape=:square)
# # plot!(slice, defz[:,9]/90, color=:grey0, label="$t9 m/s",markershape=:square)
# # plot!(slice, defz[:,10]/90, color=:firebrick, label="$t10 m/s",markershape=:square)
# # plot!(slice, defz[:,11]/90, color=:orange3, label="$t11 m/s",markershape=:square)
# # plot!(slice, defz[:,12]/90, color=:olive, markershape=:square, label="$t12 m/s", xlabel="X/D", ylabel="z*", xrotation = 0, legendtitle="Threshold", legend=:outertopright)

# # defz = defZ[:,9,:]
# # plot!(slice, defz[:,1]/90, color=:lightsalmon, label="$t1 m/s",markershape=:square)
# # plot!(slice, defz[:,2]/90, color=:orange, label="$t2 m/s",markershape=:square)
# # plot!(slice, defz[:,3]/90, color=:lightseagreen, label="$t3 m/s",markershape=:square)
# # plot!(slice, defz[:,4]/90, color=:green, label="$t4 m/s",markershape=:square)
# # plot!(slice, defz[:,5]/90, color=:dodgerblue, label="$t5 m/s",markershape=:square)
# # plot!(slice, defz[:,6]/90, color=:darkmagenta, label="$t6 m/s",markershape=:square)
# # plot!(slice, defz[:,7]/90, color=:maroon, label="$t7 m/s",markershape=:square)
# # plot!(slice, defz[:,8]/90, color=:chocolate1, label="$t8 m/s",markershape=:square)
# # plot!(slice, defz[:,9]/90, color=:grey0, label="$t9 m/s",markershape=:square)
# # plot!(slice, defz[:,10]/90, color=:firebrick, label="$t10 m/s",markershape=:square)
# # plot!(slice, defz[:,11]/90, color=:orange3, label="$t11 m/s",markershape=:square)
# # plot!(slice, defz[:,12]/90, color=:olive, markershape=:square, label="$t12 m/s", xlabel="X/D", ylabel="z*", xrotation = 0, legendtitle="Threshold", legend=:outertopright)

# # defz = defZ[:,1,:]
# # plot(slice, defz[:,1]/90, color=:lightsalmon, label="$t1 m/s",markershape=:square)
# # plot!(slice, defz[:,2]/90, color=:orange, label="$t2 m/s",markershape=:square)
# # plot!(slice, defz[:,3]/90, color=:lightseagreen, label="$t3 m/s",markershape=:square)
# # plot!(slice, defz[:,4]/90, color=:green, label="$t4 m/s",markershape=:square)
# # plot!(slice, defz[:,5]/90, color=:dodgerblue, label="$t5 m/s",markershape=:square)
# # plot!(slice, defz[:,6]/90, color=:darkmagenta, label="$t6 m/s",markershape=:square)
# # plot!(slice, defz[:,7]/90, color=:maroon, label="$t7 m/s",markershape=:square)
# # plot!(slice, defz[:,8]/90, color=:chocolate1, label="$t8 m/s",markershape=:square)
# # plot!(slice, defz[:,9]/90, color=:grey0, label="$t9 m/s",markershape=:square)
# # plot!(slice, defz[:,10]/90, color=:firebrick, label="$t10 m/s",markershape=:square)
# # plot!(slice, defz[:,11]/90, color=:orange3, label="$t11 m/s",markershape=:square)
# # plot!(slice, defz[:,12]/90, color=:olive, markershape=:square, label="$t12 m/s", xlabel="X/D", ylabel="z*", xrotation = 0, legendtitle="Threshold", legend=:outertopright)
# # savefig("smooth9.png")
# # # deflection Z threshold
# # plot(slice, defz_t[:,1]/90, color=:lightsalmon, label="$t1 m/s",markershape=:square)
# # plot!(slice, defz_t[:,2]/90, color=:orange, label="$t2 m/s",markershape=:square)
# # plot!(slice, defz_t[:,3]/90, color=:lightseagreen, label="$t3 m/s",markershape=:square)
# # plot!(slice, defz_t[:,4]/90, color=:green, label="$t4 m/s",markershape=:square)
# # plot!(slice, defz_t[:,5]/90, color=:dodgerblue, label="$t5 m/s",markershape=:square)
# # plot!(slice, defz_t[:,6]/90, color=:darkmagenta, label="$t6 m/s",markershape=:square)
# # plot!(slice, defz_t[:,7]/90, color=:maroon, markershape=:square, label="$t7 m/s", xlabel="X/D", ylabel="z*", xrotation = 0, legendtitle="Threshold", legend=:outertopright)

# # #deflection Y hybrid gauss threshold
# # plot(slice, (defy[:,1].-2500)/D, color=:lightsalmon, label="$t1 m/s",markershape=:square)
# # plot!(slice, (defy[:,2].-2500)/D, color=:orange, label="$t2 m/s",markershape=:square)
# # plot!(slice, (defy[:,3].-2500)/D, color=:lightseagreen, label="$t3 m/s",markershape=:square)
# # plot!(slice, (defy[:,4].-2500)/D, color=:green, label="$t4 m/s",markershape=:square)
# # plot!(slice, (defy[:,5].-2500)/D, color=:dodgerblue, label="$t5 m/s",markershape=:square)
# # plot!(slice, (defy[:,6].-2500)/D, color=:darkmagenta, label="$t6 m/s",markershape=:square)
# # plot!(slice, (defy[:,7].-2500)/D, color=:maroon, markershape=:square, label="$t7 m/s", xlabel="X/D", ylabel="y/D", xrotation = 0, legendtitle="Threshold", legend=:outertopright)

# # # deflection Y threshold
# # plot(slice, (defy_t[:,1].-2500)/D, color=:lightsalmon, label="$t1 m/s",markershape=:square)
# # plot!(slice, (defy_t[:,2].-2500)/D, color=:orange, label="$t2 m/s",markershape=:square)
# # plot!(slice, (defy_t[:,3].-2500)/D, color=:lightseagreen, label="$t3 m/s",markershape=:square)
# # plot!(slice, (defy_t[:,4].-2500)/D, color=:green, label="$t4 m/s",markershape=:square)
# # plot!(slice, (defy_t[:,5].-2500)/D, color=:dodgerblue, label="$t5 m/s",markershape=:square)
# # plot!(slice, (defy_t[:,6].-2500)/D, color=:darkmagenta, label="$t6 m/s",markershape=:square)
# # plot!(slice, (defy_t[:,7].-2500)/D, color=:maroon, markershape=:square, label="$t7 m/s", xlabel="X/D", ylabel="y/D", xrotation = 0, legendtitle="Threshold", legend=:outertopright)


# #sigmaz hybrid gauss threshold
# sigmaz = sigmaY[:,5,:]
# plot(slice, sigmaz[:,1], color=:lightsalmon, label="$t1 m/s",markershape=:square)
# plot!(slice, sigmaz[:,2], color=:orange, label="$t2 m/s",markershape=:square)
# plot!(slice, sigmaz[:,3], color=:lightseagreen, label="$t3 m/s",markershape=:square)
# plot!(slice, sigmaz[:,4], color=:green, label="$t4 m/s",markershape=:square)
# plot!(slice, sigmaz[:,5], color=:dodgerblue, label="$t5 m/s",markershape=:square)
# plot!(slice, sigmaz[:,6], color=:darkmagenta, label="$t6 m/s",markershape=:square)
# plot!(slice, sigmaz[:,7], color=:grey0, label="$t6 m/s",markershape=:square)
# plot!(slice, sigmaz[:,8], color=:firebrick, label="$t6 m/s",markershape=:square)
# plot!(slice, sigmaz[:,9], color=:maroon, markershape=:square, label="$t7 m/s", xlabel="X/D", ylabel="\\sigma_z/D", xrotation = 0, legendtitle="Threshold", legend=:outertopright)

# #sigmaz threshold
# plot(slice, sigmaz_t[:,1], color=:lightsalmon, label="$t1 m/s",markershape=:square)
# plot!(slice, sigmaz_t[:,2], color=:orange, label="$t2 m/s",markershape=:square)
# plot!(slice, sigmaz_t[:,3], color=:lightseagreen, label="$t3 m/s",markershape=:square)
# plot!(slice, sigmaz_t[:,4], color=:green, label="$t4 m/s",markershape=:square)
# plot!(slice, sigmaz_t[:,5], color=:dodgerblue, label="$t5 m/s",markershape=:square)
# plot!(slice, sigmaz_t[:,6], color=:darkmagenta, label="$t6 m/s",markershape=:square)
# plot!(slice, sigmaz_t[:,7], color=:maroon, markershape=:square, label="$t7 m/s", xlabel="X/D", ylabel="\\sigma_z/D", xrotation = 0, legendtitle="Threshold", legend=:outertopright)

# #sigmay hybrid gauss threshold
# plot(slice, sigmay[:,1], color=:lightsalmon, label="$t1 m/s",markershape=:square)
# plot!(slice, sigmay[:,2], color=:orange, label="$t2 m/s",markershape=:square)
# plot!(slice, sigmay[:,3], color=:lightseagreen, label="$t3 m/s",markershape=:square)
# plot!(slice, sigmay[:,4], color=:green, label="$t4 m/s",markershape=:square)
# plot!(slice, sigmay[:,5], color=:dodgerblue, label="$t5 m/s",markershape=:square)
# plot!(slice, sigmay[:,6], color=:darkmagenta, label="$t6 m/s",markershape=:square)
# plot!(slice, sigmay[:,7], color=:maroon, markershape=:square, label="$t7 m/s", xlabel="X/D", ylabel="\\sigma_y/D", xrotation = 0, legendtitle="Threshold", legend=:outertopright)

# #sigmay threshold
# plot(slice, sigmay_t[:,1], color=:lightsalmon, label="$t1 m/s",markershape=:square)
# plot!(slice, sigmay_t[:,2], color=:orange, label="$t2 m/s",markershape=:square)
# plot!(slice, sigmay_t[:,3], color=:lightseagreen, label="$t3 m/s",markershape=:square)
# plot!(slice, sigmay_t[:,4], color=:green, label="$t4 m/s",markershape=:square)
# plot!(slice, sigmay_t[:,5], color=:dodgerblue, label="$t5 m/s",markershape=:square)
# plot!(slice, sigmay_t[:,6], color=:darkmagenta, label="$t6 m/s",markershape=:square)
# plot!(slice, sigmay_t[:,7], color=:maroon, markershape=:square, label="$t7 m/s", xlabel="X/D", ylabel="\\sigma_y/D", xrotation = 0, legendtitle="Threshold", legend=:outertopright)

# plot!(slice, defz[:,8]/90, color=:chocolate1,markershape=:square, label="0.8 m/s")
# plot!(slice, defz[:,9]/90, color=:grey0,markershape=:square, label="0.9 m/s")
# plot!(slice, defz[:,10]/90, color=:firebrick,markershape=:square, label="1.0 m/s")
# plot!(slice, defz[:,11]/90, color=:blue,markershape=:square, label="1.1 m/s")
# plot!(slice, defz[:,12]/90, color=:orange3,markershape=:square, label="1.2 m/s", xlabel="X/D", ylabel="Y/HH", xrotation = 0, legendtitle="Threshold", legend=:outertopright)
# plot!(slice, defz[:,13]/90, color=:olive,markershape=:square, label="1.3 m/s",)

# savefig("deflection_SOWFA_no_negative.png")

# plot(slice, defy[:,1]/D, color=:lightsalmon, label="0.2 m/s")
# plot!(slice, defy[:,2]/D, color=:orange, label="0.3 m/s")
# plot!(slice, defy[:,3]/D, color=:lightseagreen, label="0.4 m/s")
# plot!(slice, defy[:,4]/D, color=:green, label="0.5 m/s")
# plot!(slice, defy[:,5]/D, color=:dodgerblue, label="0.6 m/s")
# plot!(slice, defy[:,6]/D, color=:darkmagenta, label="0.7 m/s")
# plot!(slice, defy[:,7]/D, color=:maroon, label="0.8 m/s", xlabel="X/D", ylabel="Y/HH", xrotation = 0, legendtitle="Threshold", legend=:outertopright)
# plot!(slice, defy[:,8]/D, color=:chocolate1, label="0.9 m/s")
# plot!(slice, defy[:,9]/D, color=:grey0, label="1.0 m/s")
# plot!(slice, defy[:,10]/D, color=:firebrick, label="1.1 m/s")
# plot!(slice, defy[:,11]/D, color=:orange3, label="1.2 m/s")
# plot!(slice, defy[:,12]/D, color=:olive, label="1.3 m/s", xlabel="X/D", ylabel="Y/HH", xrotation = 0, legendtitle="Threshold", legend=:outertopright)


# # -5 tilt (default)
# X_D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# sigmay = [0.47078852312357977, 0.6402246817896611, 0.6650094953656915, 0.5385578770834698, 0.471479784690013, 0.43605580777704167, 0.42881707103627204, 0.45662236386474814, 0.4727575411530354, 0.4952034682400052, 0.519535125091849, 0.5459527889400991,  0.5842657276970658, 0.6011867287273132, 0.6100813076639694, 0.6227672029396678, 0.6337917999159768] # sigmay
# sigmaz = [0.48987320075405877, 0.6259620168363699, 0.6551637788993473, 0.5286708690205045, 0.46534732062198364, 0.432323034144312, 0.42112711313560136, 0.4363254380684147, 0.44345326589589873, 0.4523112380723476, 0.45922940822860336, 0.46748539898758606, 0.4783548121335274, 0.4796436490639947, 0.45619902741595175, 0.5404661473623407, 0.4549516230427666]   # sigmaz
# Z_C_0 = [89.95017602408916, 95.19204606143417, 98.65202335718742, 101.07593606667089, 102.05644312661981, 103.0171313995815, 103.77931327349359, 104.82819697925221, 105.43857593425744, 106.92329294300683, 107.98603601827195, 109.61976243608325, 110.78327725658711, 112.74265461376909, 114.03172047625227, 115.59536698735086, 117.23715140026688] #Z
# Y_C_0 = [2499.098250409948, 2498.3904317175197, 2497.418197500318, 2496.23256383034, 2495.154040389873, 2493.1471288476137, 2491.1323839832835, 2489.2000126011703, 2487.2077033387136, 2485.3812163011294, 2483.4105710801496,2481.435408803965, 2479.7554485672213, 2478.347045470765, 2476.7839034380036, 2475.1385375684335, 2473.9257036151103] #Y

# X_D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# sigmay = [0.3971193430281697, 0.41076406798902254, 0.4460027078687422, 0.5933168562848211, 0.6455505766960865, 0.7256207489357791, 0.8175830014725506, 0.8212268685931307, 0.7988090894793689, 0.8206267619305435, 0.8450712997846391, 0.8294589421790787, 0.8845007578294163, 0.8769708455199341, 0.9094972317185063, 0.9041210586327585, 0.8996269381602386] # sigmay
# sigmaz = [0.386104891260254, 0.3618275632071466, 0.39500720431811637, 0.5525166220213223, 0.5940740102411888, 0.6666630928036782, 0.659959569950291, 0.6884331800051146, 0.6708600437433899, 0.6991728313728396, 0.7284330582185755, 0.716273078787434, 0.6783314804293503, 0.7124284215041349, 0.7103889153299932, 0.7072724316776328, 0.741526380661609]   # sigmaz
# Z_C_0 = [89.95017602408916, 95.19204606143417, 98.65202335718742, 101.07593606667089, 102.05644312661981, 103.0171313995815, 103.77931327349359, 104.82819697925221, 105.43857593425744, 106.92329294300683, 107.98603601827195, 109.61976243608325, 110.78327725658711, 112.74265461376909, 114.03172047625227, 115.59536698735086, 117.23715140026688] #Z
# Y_C_0 = [2499.098250409948, 2498.3904317175197, 2497.418197500318, 2496.23256383034, 2495.154040389873, 2493.1471288476137, 2491.1323839832835, 2489.2000126011703, 2487.2077033387136, 2485.3812163011294, 2483.4105710801496, 2481.435408803965, 2479.7554485672213, 2478.347045470765, 2476.7839034380036, 2475.1385375684335, 2473.9257036151103] #Y


# # 5 tilt
# X_D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# sigmay = [10.534416743706212,0.8979433225831808, 0.5569370891146653, 0.46228632687176574, 0.4418692238389031, 0.41063591760341434,0.41268488637221923,0.42743981210750065,0.4343290431766817,0.4489958375269006,0.4643980521496253 ,0.4836357967619406, 0.5010995126571905, 0.515328329804013,0.5402094497258708, 0.5632812606569052, 0.6009223174375284]   #sigy
# sigmaz = [] #sigz
# Z_C_5 = [] #Z
# Y_C_5 = []  #Y


## 10 tilt
# X_D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# sigmay = [11.566856354862932, 0.8185251072516069,0.5355890686338116, 0.45141676234400213, 0.4198328468956569,0.41092625935179605, 0.4365946044526015, 0.45786695605186933, 0.4761876253994781,  0.5024869779943413, 0.5432325346790041, 0.5607871134207331, 0.592533332663219, 0.6477570021530344, 0.696919810604235, 0.7424828574881366, 0.7963861251101604] # Y
# sigmaz = []   # Z
# Z_C_10 = [] #Z
# Y_C_10 = []  #Y


## 25 tilt
# Noticeable kidney bean shape around 10 D downstream
# X_D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# sigmay = [1.0486518586374045, 0.929293481327818, 0.5013492046906606,  0.4703943454181189, 0.5141311666540866, 0.5227016949321093, 0.571273034792967, 0.6984934204639048, 0.8170952006429221, 0.9800922833172495, 1.1046498554300472, 1.261702986758678, 1.38654058454812, 1.6379134114473277, 1.8018044888183442, 2.007842886875235, 2.1578047721042277] #sigy
# sigmaz = []  #sigz
# Z_C_25 = [] #Z
# Y_C_25 = []  #Y



## -15 tilt
# X_D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# sigmay = [1.6079674017100833, 2.0058850864305193, 0.6464026239523623, 0.4982147063609224, 0.4298866662767064, 0.3790282034525116,0.36563170476511053, 0.37339419800048723, 0.3487206493838459, 0.35726981422395804, 0.3594724581663881, 0.33551358029123796, 0.3283727700109541, 0.30980429628409684, 0.26902110279114033, 0.23748843617942725, 0.2197265098782207] #sigy
# sigmaz = [0.949429050435477, 1.6618690680233767, 0.48852696359525305, 0.38868326448953006, 0.35269031328577893, 0.3177947941831356, 0.31977499921936153, 0.333627422131592, 0.35465150192559197, 0.3919192425454063, 0.4070657577618838, 0.44866105891282876, 0.4845735336212371, 0.5241963213842247, 0.5530000112319885, 0.6265598774479376, 0.6619233506141613]  #sigz
# Z_C_15_n = [88.15958593107578, 98.23708436457522, 104.51298129779424, 110.13933665936386, 113.81765574654712, 117.13170636290126, 120.13255946747951, 122.52972220105784, 124.24880477273803, 126.52268793026317, 127.91193648747887, 128.9873689824886, 129.5293452912365, 130.74622516535294, 131.3151812026898, 131.97994529732821, 132.70137331226812] #Z
# Y_C_15_n = [2499.458080542129, 2497.326590321634, 2497.3707534727528, 2495.0208774988682, 2493.3587635138374, 2490.605822750391, 2489.0959217663976, 2486.3232236375966, 2484.677127601138, 2481.63237944456, 2480.869684054634, 2478.818191956405, 2476.728508334095, 2475.68174855053, 2474.8342232757345, 2473.8641488032813, 2473.473876998827]  #Y


## -20 tilt
# X_D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# sigmay = [1.2552426767761358, 1.7803897667598843, 0.6463062709533344, 0.4921535563291503, 0.4233110506240444, 0.3969723445537178, 0.379830997727377, 0.3771159496188053, 0.38169479447061555, 0.3562513068381116, 0.35799526884431887, 0.3500638518415055, 0.3277814394102879, 0.2810343469258474, 0.22279678971917802, 0.18127615972200717, 0.10657345360719093] #sigy
# sigmaz = [0.6669235730726144, 1.2148165661261072, 0.4765130372073104, 0.3912837625878571, 0.3503571080300599, 0.3408815793355835, 0.3236542355944509, 0.339051853526612, 0.37697531328877676, 0.4132354402859387, 0.46658483758661434, 0.5126366084068461, 0.5578363947398289, 0.6017349645383988, 0.6354914143920146, 0.6487096419945122, 0.737022663190537]  #sigz
# Z_C_20_n = [86.67967290270168, 99.57409026516494, 106.9731067916532, 113.17082138162434, 117.68569837774646, 121.10515613116911, 125.0647932743461, 127.59230194192958, 130.0875399030215, 132.60539960368172, 134.50330716038027, 136.9624073411914, 138.21897184021145, 139.670846878197, 141.41488854517448, 142.9225867769201, 143.48364527942243] #Z
# Y_C_20_n = [2499.5017460005624, 2499.18071915521, 2496.0314264539966, 2494.209479744589, 2494.177134794114, 2490.1312412042626, 2487.7475387888894, 2485.6283524518453, 2483.3461286702814, 2481.901625030293, 2480.214783617409, 2477.351825323099, 2475.2401820779696, 2473.9938396100483, 2471.382073553874, 2469.9704467611814, 2470.10531865683]  #Y

## -35 tilt
# X_D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# sigmay = [0.8244189698800157, 0.6024029799815012, 0.6093224081303246, 0.44960976022089494, 0.3930630568303988, 0.3981832174820402, 0.40878289334383894, 0.4133568239744189, 0.4316356803488093, 0.44549026593433294, 0.41397123983424244, 0.3373843700209892, 0.24227933603634463, 0.16628784154209447, 0.08308782524573832, 0.04155571896672743, 0.0003380983183455808] #sigy
# sigmaz = [0.19442217786655325, 0.28849874304008444, 0.33692615612725063, 0.2884963264451269, 0.2528694925146376, 0.2702897752485132, 0.2999455599959315, 0.3271669864639892, 0.408422978188509, 0.49748974732356116, 0.5317605069322571, 0.5914962077570648, 0.6115388247913511, 0.6766985283865756, 0.6850189280875735, 0.7212086761124722, 0.777435658398414]  #sigz
# Z_C_35_n = [86.24157466564435, 101.07226492136716, 107.98423990611295, 114.48634452501115, 119.33816986028027, 123.96146024920974, 127.60961005557837, 130.7080488408067, 133.7541541174053, 136.09111053613393, 137.20979265905464, 139.16941083706243, 140.03357910572657, 142.18861329197557, 143.21457516698212, 145.516263445937, 147.54833712565144 ] #Z
# Y_C_35_n = [2499.4479240528767, 2497.5936804928006, 2497.5560261660526, 2494.3777931940303, 2490.980027067911, 2489.7095444419933, 2487.9339418725854, 2484.716542663103, 2483.3452186122245, 2480.4934846644182, 2477.110726834622, 2475.8103683288755, 2473.497260000254, 2471.422293511948, 2468.8066275355054, 2466.756811846001, 2466.9068485137072]  #Y



"""Sigma data"""
# How are these dervied?
# sigma0x = 0.29874
# sigma0y = 0.15441985668899924

# # -15 tilt
# X_D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# sigmay = [1.6079674017100833, 2.0058850864305193, 0.6464026239523623, 0.4982147063609224, 0.4298866662767064, 0.3790282034525116,0.36563170476511053, 0.37339419800048723, 0.3487206493838459, 0.35726981422395804, 0.3594724581663881, 0.33551358029123796, 0.3283727700109541, 0.30980429628409684, 0.26902110279114033, 0.23748843617942725, 0.2197265098782207] #sigy
# sigmaz = [0.949429050435477, 1.6618690680233767, 0.48852696359525305, 0.38868326448953006, 0.35269031328577893, 0.3177947941831356, 0.31977499921936153, 0.333627422131592, 0.35465150192559197, 0.3919192425454063, 0.4070657577618838, 0.44866105891282876, 0.4845735336212371, 0.5241963213842247, 0.5530000112319885, 0.6265598774479376, 0.6619233506141613]  #sigz
# Z_C_15_n = [88.15958593107578, 98.23708436457522, 104.51298129779424, 110.13933665936386, 113.81765574654712, 117.13170636290126, 120.13255946747951, 122.52972220105784, 124.24880477273803, 126.52268793026317, 127.91193648747887, 128.9873689824886, 129.5293452912365, 130.74622516535294, 131.3151812026898, 131.97994529732821, 132.70137331226812] #Z
# Y_C_15_n = [2499.458080542129, 2497.326590321634, 2497.3707534727528, 2495.0208774988682, 2493.3587635138374, 2490.605822750391, 2489.0959217663976, 2486.3232236375966, 2484.677127601138, 2481.63237944456, 2480.869684054634, 2478.818191956405, 2476.728508334095, 2475.68174855053, 2474.8342232757345, 2473.8641488032813, 2473.473876998827]  #Y


# # -20 tilt
# X_D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# sigmay = [1.2552426767761358, 1.7803897667598843, 0.6463062709533344, 0.4921535563291503, 0.4233110506240444, 0.3969723445537178, 0.379830997727377, 0.3771159496188053, 0.38169479447061555, 0.3562513068381116, 0.35799526884431887, 0.3500638518415055, 0.3277814394102879, 0.2810343469258474, 0.22279678971917802, 0.18127615972200717, 0.10657345360719093] #sigy
# sigmaz = [0.6669235730726144, 1.2148165661261072, 0.4765130372073104, 0.3912837625878571, 0.3503571080300599, 0.3408815793355835, 0.3236542355944509, 0.339051853526612, 0.37697531328877676, 0.4132354402859387, 0.46658483758661434, 0.5126366084068461, 0.5578363947398289, 0.6017349645383988, 0.6354914143920146, 0.6487096419945122, 0.737022663190537]  #sigz
# Z_C_20_n = [86.67967290270168, 99.57409026516494, 106.9731067916532, 113.17082138162434, 117.68569837774646, 121.10515613116911, 125.0647932743461, 127.59230194192958, 130.0875399030215, 132.60539960368172, 134.50330716038027, 136.9624073411914, 138.21897184021145, 139.670846878197, 141.41488854517448, 142.9225867769201, 143.48364527942243] #Z
# Y_C_20_n = [2499.5017460005624, 2499.18071915521, 2496.0314264539966, 2494.209479744589, 2494.177134794114, 2490.1312412042626, 2487.7475387888894, 2485.6283524518453, 2483.3461286702814, 2481.901625030293, 2480.214783617409, 2477.351825323099, 2475.2401820779696, 2473.9938396100483, 2471.382073553874, 2469.9704467611814, 2470.10531865683]  #Y

# # -35 tilt
# X_D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# sigmay = [0.8244189698800157, 0.6024029799815012, 0.6093224081303246, 0.44960976022089494, 0.3930630568303988, 0.3981832174820402, 0.40878289334383894, 0.4133568239744189, 0.4316356803488093, 0.44549026593433294, 0.41397123983424244, 0.3373843700209892, 0.24227933603634463, 0.16628784154209447, 0.08308782524573832, 0.04155571896672743, 0.0003380983183455808] #sigy
# sigmaz = [0.19442217786655325, 0.28849874304008444, 0.33692615612725063, 0.2884963264451269, 0.2528694925146376, 0.2702897752485132, 0.2999455599959315, 0.3271669864639892, 0.408422978188509, 0.49748974732356116, 0.5317605069322571, 0.5914962077570648, 0.6115388247913511, 0.6766985283865756, 0.6850189280875735, 0.7212086761124722, 0.777435658398414]  #sigz
# Z_C_35_n = [86.24157466564435, 101.07226492136716, 107.98423990611295, 114.48634452501115, 119.33816986028027, 123.96146024920974, 127.60961005557837, 130.7080488408067, 133.7541541174053, 136.09111053613393, 137.20979265905464, 139.16941083706243, 140.03357910572657, 142.18861329197557, 143.21457516698212, 145.516263445937, 147.54833712565144 ] #Z
# Y_C_35_n = [2499.4479240528767, 2497.5936804928006, 2497.5560261660526, 2494.3777931940303, 2490.980027067911, 2489.7095444419933, 2487.9339418725854, 2484.716542663103, 2483.3452186122245, 2480.4934846644182, 2477.110726834622, 2475.8103683288755, 2473.497260000254, 2471.422293511948, 2468.8066275355054, 2466.756811846001, 2466.9068485137072]  #Y


# # -5 tilt (default)
# X_D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# sigmay = [3.620652395011077, 1.9001187174615162, 0.9353771135750221, 0.5107471971198718, 0.42978473208759005, 0.3839338461743554, 0.374063788603435, 0.39012705439447876, 0.38695709154680563, 0.39829126421577415, 0.39095118850320143, 0.39704689354359385, 0.3977004627441319, 0.4107123220098429, 0.4129081941055791, 0.4344748292590568, 0.4337127025518822] # sigmay
# sigmaz = [2.2432098786983983, 1.3961719125350165, 0.755954254565374, 0.4592818994838727, 0.40318738840385854, 0.37372102633918103, 0.367752253933641, 0.3744143075411739, 0.38718786846898806, 0.3981052800029568, 0.4121418848254266, 0.4274035829902239, 0.4388200797964752, 0.4295260071830763, 0.4398919361685257, 0.4456690402908989, 0.45604487734391563]   # sigmaz
# Z_C_0 = [89.94610061435328, 95.49645626896056, 98.2265475064328, 100.49529508064013, 102.72240537093262, 103.85038109397676, 104.23645245320908, 105.01900208844084, 104.93112639911536, 104.7918075084111, 104.9938387356675, 104.82326870778621, 104.86740609339417, 104.83345927184934, 104.69344627371628, 104.80461048081625, 104.35492208525451] #Z
# Y_C_0 = [2499.523949696036, 2497.508003392592, 2497.6743370066906, 2495.863164631482, 2495.159256668798, 2493.451633868584, 2490.88406339492, 2490.7839345469697, 2487.6064300432463, 2485.859849423469, 2484.6473931202268, 2481.7582190813796, 2481.5359719974263, 2480.7214458760063, 2480.3737585938, 2478.883566974432, 2478.468778649857] #Y


# # 5 tilt
# X_D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# sigmay = [10.534416743706212,0.8979433225831808, 0.5569370891146653, 0.46228632687176574, 0.4418692238389031, 0.41063591760341434,0.41268488637221923,0.42743981210750065,0.4343290431766817,0.4489958375269006,0.4643980521496253 ,0.4836357967619406, 0.5010995126571905, 0.515328329804013,0.5402094497258708, 0.5632812606569052, 0.6009223174375284]   #sigy
# sigmaz = [6.350511686246529, 0.6520098397072024, 0.43916231513688375, 0.3746798584886675,0.3602545016781201, 0.3226843978434129, 0.3080967463394823,0.3039904260707824,0.3005203899809468,0.29732923038334,0.2986499019389889 ,0.2965946573010577, 0.2910329829458407, 0.2901500696761998, 0.2901309224379686, 0.2891634120433182, 0.296594048549694] #sigz
# Z_C_5 = [89.09719716901935, 90.83767756026221, 90.57815228052672, 90.56968791075349, 89.84733116085401, 88.90706483595609, 87.81384325102103, 86.17736323455676, 84.93916831923703, 83.33615473808837, 83.01225858048416, 83.01225858048416, 83.36245394470659, 83.49590653288737, 83.75392978996629, 83.53494826586216, 83.78266554390308] #Z
# Y_C_5 = [2498.844542166793, 2498.472154616501, 2498.642610484224, 2496.735668172087, 2497.458928903369, 2494.971424293041, 2495.233783501523, 2491.9747863764346, 2490.42943217409, 2488.5478994637215, 2486.912432107493, 2485.843154252683, 2485.4458719136783, 2483.975377615859, 2481.833086416161, 2479.6583534584865, 2478.742380819284]  #Y


# # 10 tilt
# X_D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# sigmay = [11.566856354862932, 0.8185251072516069,0.5355890686338116, 0.45141676234400213, 0.4198328468956569,0.41092625935179605, 0.4365946044526015, 0.45786695605186933, 0.4761876253994781,  0.5024869779943413, 0.5432325346790041, 0.5607871134207331, 0.592533332663219, 0.6477570021530344, 0.696919810604235, 0.7424828574881366, 0.7963861251101604] # Y
# sigmaz = [6.783853290532606, 0.5910436850172545, 0.4338064559726529, 0.35225416739924853, 0.30711028779394,  0.27819983503607737, 0.25761052244449867, 0.24702471526565073, 0.23594624795837132,0.22987186516455954,0.2270829576046232, 0.2199480212623694, 0.21557963364859323, 0.2137683702248472, 0.20958322679802785, 0.2025967047878765, 0.20408758263581525]   # Z
# Z_C_10 = [89.84930372573861, 89.15685807798387, 87.95254645572378, 86.93113236227126, 85.36731160248235, 84.22435662501483, 83.02633897451912, 81.51882521152714, 80.37198214327715, 80.15840546542358, 79.88989596041188, 79.9248167898811, 80.63345656385897, 81.45434499142415, 81.77754154892244, 82.67950937818864, 83.29039704849045] #Z
# Y_C_10 = [2498.952574389726, 2497.6993665868113, 2499.0557606436373, 2497.970465656208, 2497.7687994141324, 2496.5773441608903, 2495.550145160467, 2492.2674099726833, 2491.995153619506, 2489.8344180040776, 2488.8719334679317, 2488.572397809847, 2486.5378318959324, 2485.032066740046, 2482.946141210619, 2478.99808802504, 2475.3402585202098]  #Y


# # 25 tilt
# # Noticeable kidney bean shape around 10 D downstream
# X_D = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# sigmay = [1.0486518586374045, 0.929293481327818, 0.5013492046906606,  0.4703943454181189, 0.5141311666540866, 0.5227016949321093, 0.571273034792967, 0.6984934204639048, 0.8170952006429221, 0.9800922833172495, 1.1046498554300472, 1.261702986758678, 1.38654058454812, 1.6379134114473277, 1.8018044888183442, 2.007842886875235, 2.1578047721042277] #sigy
# sigmaz = [0.4046093385523854, 0.5511559633520392, 0.26678936599278036, 0.21017174971273248, 0.16734336575795677, 0.12584344241611423, 0.09809101432374921, 0.06939870802606617, 0.046205874802820276, 0.017427648446448302, -0.008797192917793782, -0.03504928798436168, -0.05928478605242399, -0.08988956947543056, -0.11409218319420603, -0.14206048303716606, -0.16536844136511167]  #sigz
# Z_C_25 = [90.56968249109853, 82.16694970360412, 77.67281363265396, 73.2499960248699, 70.46213068067172, 67.28112760800242, 63.830002769756256, 62.58413771830426, 60.84087272998218, 60.74035480174294, 60.50779317293857, 61.29980418883348, 61.50922354059292, 62.24987121347377, 62.54121663872897, 63.34436655991553, 63.28341317975336] #Z
# Y_C_25 = [2499.2230841068795, 2499.6188773649087, 2498.801895435778, 2497.976160591793, 2499.435063879633, 2498.5479741789864, 2495.763863434187, 2493.920857733599, 2494.040014654551, 2495.147623157222, 2497.269334989691, 2495.4982137558186, 2494.7547797690745, 2489.943247887391, 2489.0370058659, 2477.0301802742542, 2456.8345831043994]  #Y


# # polyfit for each angle
# endz = 16
# start35 = 5
# start20 = 7
# start15 = 6
# start0 = 6
# start5 = 6
# start10 = 6
# start25 = 5
# # -5
# beginning = 1
# Z_values_0, Y_values_0, ky_0, kz_0, z0_0, y0_0 = kykzint(sigmay_0[start0:endz], sigmaz_0[start0:endz], X_D[start0:endz], X_D[start0:endz], beginning)

# # 5
# beginning = 1
# Z_values_5, Y_values_5, ky_5, kz_5, z0_5, y0_5 = kykzint(sigmay_5[start5:endz], sigmaz_5[start5:endz], X_D[start5:endz], X_D[start5:endz], beginning)

# # -5
# beginning = 1
# Z_values_10, Y_values_10, ky_10, kz_10, z0_10, y0_10 = kykzint(sigmay_10[start10:endz], sigmaz_10[start10:endz], X_D[start10:endz], X_D[start10:endz], beginning)

# # -5
# beginning = 1
# Z_values_25, Y_values_25, ky_25, kz_25, z0_25, y0_25 = kykzint(sigmay_25[start25:endz], sigmaz_25[start25:endz], X_D[start25:endz], X_D[start25:endz], beginning)

# # -15
# beginning = 1
# Z_values_15_n, Y_values_15_n, ky_15_n, kz_15_n, z0_15_n, y0_15_n = kykzint(sigmay_15_n[start15:endz], sigmaz_15_n[start15:endz], X_D[start15:endz], X_D[start15:endz], beginning)

# # -20
# beginning = 1
# Z_values_20_n, Y_values_20_n, ky_20_n, kz_20_n, z0_20_n, y0_20_n = kykzint(sigmay_20_n[start20:endz], sigmaz_20_n[start20:endz], X_D[start20:endz], X_D[start20:endz], beginning)

# # -35
# beginning = 1
# Z_values_35_n, Y_values_35_n, ky_35_n, kz_35_n, z0_35_n, y0_35_n = kykzint(sigmay_35_n[start35:endz], sigmaz_35_n[start35:endz], X_D[start35:endz], X_D[start35:endz], beginning)



# # plotting
# # # sigmaz
# endz = 16


# plot(X_D[start35:endz], sigmaz_35_n[start35:endz],seriestype=:scatter,markershape=:square,color=:orange,  label="\\gamma=-35")
# plot!(X_D[start20:endz], sigmaz_20_n[start20:endz],seriestype=:scatter,markershape=:star,color=:purple, label="\\gamma=-20")
# plot!(X_D[start15:endz], sigmaz_15_n[start15:endz],seriestype=:scatter,markershape=:star5,color=:magenta,  label="\\gamma=-15")
# plot!(X_D[start0:endz], sigmaz_0[start0:endz],seriestype=:scatter,markershape=:circle,color=:red, label="\\gamma=-5")
# plot!(X_D[start5:endz], sigmaz_5[start5:endz],seriestype=:scatter,markershape=:square,color=:blue,  label="\\gamma=5")
# plot!(X_D[start10:endz], sigmaz_10[start10:endz],seriestype=:scatter,markershape=:star,color=:green, label="\\gamma=10")
# plot!(X_D[start25:endz], sigmaz_25[start25:endz],seriestype=:scatter,markershape=:star5,color=:black,  label="\\gamma=25")
# plot!(X_D[start35:endz], Y_values_35_n,color=:orange,label="kz=$kz_35_n")
# plot!(X_D[start20:endz], Y_values_20_n,color=:purple,label="kz=$kz_20_n")
# plot!(X_D[start15:endz], Y_values_15_n,color=:magenta, label="kz=$kz_15_n")
# plot!(X_D[start0:endz], Y_values_0,color=:red,label="kz=$kz_0")
# plot!(X_D[start5:endz], Y_values_5,color=:blue,label="kz=$kz_5")
# plot!(X_D[start10:endz], Y_values_10,color=:green, label="kz=$kz_10")
# plot!(X_D[start25:endz], Y_values_25,color=:black,label="kz=$kz_25", ylabel="\\sigma_z/D", xrotation = 0, xlabel="X/D", legend=:outertopright, grid=false)
# # xlims!(3.8,11.2)
# # ylims!(-0.22,0.6)
# savefig("kz_alltilt_neg_pos_new.png")

# # # # sigmay
# plot(X_D[start35:endz], sigmay_35_n[start35:endz],seriestype=:scatter,markershape=:square,color=:orange,  label="\\gamma=-35")
# plot(X_D[start20:endz], sigmay_20_n[start20:endz],seriestype=:scatter,markershape=:star,color=:purple, label="\\gamma=-20")
# plot!(X_D[start15:endz], sigmay_15_n[start15:endz],seriestype=:scatter,markershape=:star5,color=:magenta,  label="\\gamma=-15")
# plot!(X_D[start0:endz], sigmay_0[start0:endz],seriestype=:scatter,markershape=:circle,color=:red, label="\\gamma=-5")
# plot!(X_D[start5:endz], sigmay_5[start5:endz],seriestype=:scatter,markershape=:square,color=:blue,  label="\\gamma=5")
# plot!(X_D[start10:endz], sigmay_10[start10:endz],seriestype=:scatter,markershape=:star,color=:green, label="\\gamma=10")
# plot!(X_D[start25+1:endz], sigmay_25[start25+1:endz],seriestype=:scatter,markershape=:star5,color=:black,  label="\\gamma=25")
# # plot!(X_D[start35:endz], Z_values_35_n,color=:orange,label="kz=$kz_35_n")
# plot!(X_D[start20:endz], Z_values_20_n,color=:purple,label="ky=$kz_20_n")
# plot!(X_D[start15:endz], Z_values_15_n,color=:magenta, label="ky=$kz_15_n")
# plot!(X_D[start0:endz], Z_values_0,color=:red,label="ky=$kz_0")
# plot!(X_D[start5:endz], Z_values_5,color=:blue,label="ky=$kz_5")
# plot!(X_D[start10:endz], Z_values_10,color=:green, label="ky=$kz_10")
# plot!(X_D[start25+1:endz], Z_values_25[2:end],color=:black,label="ky=$kz_25", ylabel="\\sigma_y/D", xrotation = 0, xlabel="X/D", legend=:outertopright, grid=false)
# # xlims!(3.8,11.2)
# # ylims!(-0.22,0.6)
# savefig("ky_alltilt_new_pos_new.png")













# # # ky and kz as a function of tilt
# Tilt = [-5, 5, 10]
# Tilty = [-5, 5, 10]
# ky_arr = [0.018, 0.027, 0.03]
# kz_arr = [0.025, 0.014, 0.017]
# kz_arr_d = [-0.002, 0.006, -0.002]

# Q = polyfit(Tilty, ky_arr, 1)
# ky_val = Q[1]
# eps_y = Q[0]
# Z_values = eps_y.+ky_val.*Tilty

# Q = polyfit(Tilt, kz_arr, 1)
# kz_val = Q[1]
# eps_z = Q[0]
# Y_values = eps_z.+kz_val.*Tilt

# Q = polyfit(Tilt, kz_arr_d, 1)
# kz_val_d = Q[1]
# eps_z_d = Q[0]
# Y_values_d = eps_z_d.+kz_val_d.*Tilt

# ky_val = round(ky_val, digits=5)
# kz_val = round(kz_val, digits=5)
# kz_val_d = round(kz_val_d, digits=5)
# eps_y = round(eps_y, digits=5)
# eps_z = round(eps_z, digits=5)
# eps_z_d = round(eps_z_d, digits=5)
# plot(Tilty, ky_arr, seriestype=:scatter,color=:blue, label="ky")
# plot!(Tilt, kz_arr, seriestype=:scatter,color=:red, label="kz upper")
# plot!(Tilt, kz_arr_d, seriestype=:scatter,color=:green, label="kz lower")
# plot!(Tilty, Z_values,label="$ky_val\\gamma + $eps_y", color=:blue)
# plot!(Tilt, Y_values,label="$kz_val\\gamma + $eps_z", color=:red)
# plot!(Tilt, Y_values_d,label="$kz_val_d\\gamma + $eps_z_d", color=:green, legend=:left, ylabel="k-value", xrotation = 0, xlabel="Turbine Tilt (degrees)", grid=false)
# savefig("kykztilt_both_new.png")














# -5 tilt (default)
# X_D = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# sigmay = [0.43558933580700243, 0.3849023106472279, 0.371602317089002, 0.3784555734334545, 0.384645377630294, 0.3821693621079923, 0.3920122241892837, 0.40906682219417306] # sigmay
# sigmaz = [0.3248600444941308, 0.28934389948149897, 0.27230179133941784, 0.2637512382064413, 0.23999184075039431, 0.22293325144256623, 0.2111828753220627, 0.19827940294790886]   # sigmaz
# Z_C_0 = [99.63967017354084, 98.45427852648643, 96.59368232105771, 91.95162426180555, 85.0814652582951, 76.46572702298704, 69.9175617140443, 60.266311677827595] #Z
# Y_C_0 = [2497.0221038268683, 2495.4384888802556, 2494.2258039351427, 2494.6278563934866, 2494.730346059316, 2496.09527267563, 2496.8996770145145, 2490.382155887092] #Y


# X_D = [4, 5, 6, 7, 8, 9, 10, 11]
# sigmay = [0.43558933580700243, 0.38682713173418865, 0.3729651664824919, 0.3820953498643966, 0.39477694638407024, 0.395521043066644, 0.40506571281243925, 0.410193929282697, 0.4190685603089271, 0.42548378944096543, 0.42577196393967665] #Y
# sigmaz = [0.3248600444941308, 0.30429338285686497, 0.28678867463386587, 0.28051406001833296, 0.27548699524201975, 0.28768974733321134, 0.2823458674900679, 0.2714577088623376, 0.261623344290019, 0.2513207565201982, 0.2121673077529078]    #Z
# Z_C_0 = [99.76862781602301, 103.50046574468377, 105.86697482403154, 107.78640388442014, 109.72850601128532, 110.3134873081655, 110.33635752886255, 110.61448355053594, 110.03282647975072, 110.3151921530234, 109.60895778654945] #Z
# Y_C_0 = [2496.118755976693, 2496.4528605178893, 2496.059090971596, 2495.7312629558537, 2496.569377119317, 2496.720327553422, 2496.7517153476315, 2496.3985982537256,  2496.35324849247, 2496.8414607376394, 2496.4207757132504] # dY

# 5 tilt
# X_D = [4, 5, 6, 7, 8, 9, 10, 11]
# sigmay = [0.43768156977770806, 0.4008992621275602, 0.3985506784434525, 0.40588138933769524, 0.4213990843170151, 0.43935841167912815, 0.4503615462538922, 0.48473136277857326]
# sigmaz = [0.2827904780501523, 0.24403377341394164, 0.22004936957576926, 0.20288644282846177, 0.17262718039501648, 0.14632905780823344, 0.12709286395116817, 0.10233420322283424]
# Z_C_5 = [88.4061997235887, 87.6668665653627, 83.53242730274529, 80.02195980208653, 73.8526132629067, 67.81575780501393, 63.00202558179132, 57.41052962659714] #Z
# Y_C_5 = [2498.1670258309923, 2497.387464043016, 2497.6080703069997, 2496.064361518402, 2496.1784027004824, 2496.8795619057587, 2495.2518755845895, 2494.299918234909]  #Y

# 10 tilt
# X_D = [4, 5, 6, 7, 8, 9, 10, 11]
# sigmay = [0.45806267486976093, 0.39692834627417295, 0.41135144280690256, 0.4253305807374344, 0.44906416444232894, 0.46475184784103357, 0.49581184974891784, 0.5106992676959436] # Y
# sigmaz = [0.2243011134004611, 0.19216694758207092, 0.1640563922780164, 0.14210369170699114, 0.1127625035202978, 0.0907907337176842, 0.06386642457900364,  0.044683091016937175]   # Z
# Z_C_10 = [83.49768448270565, 81.74066544888844, 76.83889812534574, 71.6312013311421, 67.3194618675204, 63.20050680722589, 58.25342380244782, 53.2284885542085] #Z
# Y_C_10 = [2500.202643252283, 2498.305648161068, 2498.696901955447, 2498.1955580107924, 2497.253076009381, 2496.4693414156313, 2497.8010114824424, 2497.54738890988]  #Y

# 25 tilt
# X_D = [4, 5, 6, 7, 8, 9, 10, 11]
# sigmay = [0.4727635239666252, 0.47105538761539123, 0.5014435044827972, 0.5278467146454514, 0.5640883377715035, 0.5860256685796438, 0.6121778442264209, 0.6409016657181129]
# sigmaz = [0.07012474670578521, 0.026754681513178256, -0.013072929973284412, -0.050440626623954106,  -0.09496277194798287, -0.12211945349343398, -0.16239398469050798, -0.20609922972605751]
# Z_C_25 = [67.25371769433993, 62.130306440546086, 58.4666343552621, 55.25496964164948, 51.66377112780846, 48.50200230182843, 44.92602406185611, 41.753554873174146] #Z
# Y_C_25 = [2499.6173322272584, 2499.643499264882, 2498.9038939934544, 2498.180105592254, 2498.149297223707, 2495.567860476737, 2496.69270996085, 2498.2997845054474]  #Y




# seriestype=:scatter,markershape=:square,color=:orange
# seriestype=:scatter,markershape=:star,color=:purple
# seriestype=:scatter,markershape=:star5,color=:magenta
# seriestype=:scatter,markershape=:circle,color=:red
# seriestype=:scatter,markershape=:square,color=:blue
# seriestype=:scatter,markershape=:star,color=:green
# seriestype=:scatter,markershape=:star5,color=:black
# Plot deflections
# plot(X_D, (Y_C_35_n.-2500)/D, label="-35", seriestype=:scatter,markershape=:square,color=:orange)
# plot!(X_D, (Y_C_20_n.-2500)/D, label="-20", seriestype=:scatter,markershape=:star,color=:purple)
# plot!(X_D, (Y_C_15_n.-2500)/D, label="-15", seriestype=:scatter,markershape=:star5,color=:magenta)
# plot!(X_D, (Y_C_0.-2500)/D, label="-5", seriestype=:scatter,markershape=:circle,color=:red)
# plot!(X_D, (Y_C_5.-2500)/D, label="5", seriestype=:scatter,markershape=:square,color=:blue)
# plot!(X_D, (Y_C_10.-2500)/D, label="10", seriestype=:scatter,markershape=:star,color=:green)
# plot!(X_D, (Y_C_25.-2500)/D, label="25",seriestype=:scatter,markershape=:star5,color=:black,  ylabel="\\delta_y/D", xrotation = 0, xlabel="X/D", legendtitle="Turbine Tilt", legend=:bottom, grid=false)
# xlims!((0,12))
# savefig("deflection_y.png")

# plot(X_D, (Z_C_35_n)/90, label="-35", seriestype=:scatter,markershape=:square,color=:orange)
# plot!(X_D, (Z_C_20_n)/90, label="-20", seriestype=:scatter,markershape=:star,color=:purple)
# plot!(X_D, (Z_C_15_n)/90, label="-15", seriestype=:scatter,markershape=:star5,color=:magenta)
# plot!(X_D, (Z_C_0)/90, label="-5", seriestype=:scatter,markershape=:circle,color=:red)
# plot!(X_D, (Z_C_5)/90, label="5", seriestype=:scatter,markershape=:square,color=:blue)
# plot!(X_D, (Z_C_10)/90, label="10", seriestype=:scatter,markershape=:star,color=:green)
# plot!(X_D, (Z_C_25)/90, label="25",seriestype=:scatter,markershape=:star5,color=:black,  ylabel="\\delta_z*", xrotation = 0, xlabel="X/D", legendtitle="Turbine Tilt", legend=:outertopright, grid=false)
# xlims!((0,12))
# savefig("deflection_Z.png")


# # data1 = heatmap(Y_plot, Y_plot, clim=(0,9), FLORIS_data)
# # plot(data1, ylim=(0,300), xlim=(700,1150), aspect_ratio=:equal)



# # Equation 4.1 in Banstkanhah wake model paper (2016)
# sigma_z_d_comp = -((1/(sqrt(2*pi)*(U_c_vert)))*-trapz(crop_Z_plot_comp,(U_c_vert_p)))/D
# sigma_y_d_comp = -((1/(sqrt(2*pi)*(U_c_horz)))*-trapz(crop_Y_plot_comp[Sigma_y_crop[1]:Sigma_y_crop[2]],(U_c_horz_p[Sigma_y_crop[1]:Sigma_y_crop[2]])))/D

# # Z_avg
# # use trapz
# # Area = 

# """Find center location and respective velocity"""
# # Basntankhah paper sepcifices that minimum velocity deficit is where center of wake is
# # This assumption is true past about 5D downstream of the turbine
# min_loc = argmin(SOWFA_data)
# U_c = SOWFA_data[min_loc[1],min_loc[2]]


# # """plot cropped data and min location"""
# data1 = heatmap(crop_Y_plot_comp, crop_Z_plot_comp, cropped_SOWFA_data_comp)
# plot(data1, aspect_ratio=:equal)

# # data1 = heatmap(crop_Y_plot_comp, crop_Z_plot_comp, negative)
# # plot(data1, aspect_ratio=:equal)

# # data1 = heatmap(crop_Y_plot_comp, crop_Z_plot_comp,cropped_SOWFA_data_comp_neg)
# # plot(data1, aspect_ratio=:equal)
# # plot!([crop_Y_plot[min_loc[1]]], [crop_Z_plot[min_loc[2]]], color="blue")
# # scatter!([crop_Y_plot[min_loc[2]]], [crop_Z_plot[min_loc[1]]], color="blue", label="wake center")
# print("Y_cm: ", Y_cm_comp, "\n")
# print("Z_cm: ", Z_cm_comp)
# scatter!(crop_Y_plot_comp[Y_whole_wake_comp], crop_Z_plot_comp[X_whole_wake_comp], color="red", label="Wake Shape")
# scatter!([2500.0], [90.0], color="blue", label="center of turbine")
# scatter!([Y_cm_comp], [Z_cm_comp], color="green", label="Center of Mass")
# xlims!((2250,2750))
# # # data1 = heatmap(Y_plot, Y_plot, clim=(0,9), FLORIS_data)
# # # plot(data1, ylim=(0,300), xlim=(700,1150), aspect_ratio=:equal)
# print("sig z: ", sigma_z_d_comp, "\n")
# print("sig y: ", sigma_y_d_comp, "\n")




# """Looking at finding relationship for sig0y and sig0z"""
# # SOWFA
# sig0y = [0.6324600220803214, 0.5420122398617231, 0.45555274142164887, 0.35337147482723696, 0.3178969525976916, 0.22800268824662198, -0.2302104789208812]
# sig0z = [0.040171620571319316, 0.06732198572068403, 0.1269824071340052, 0.3225851494794991, 0.32627652890037356, 0.29616027651514, 0.2650910334989699]

# TILT = [-35, -20, -15, -5, 5, 10, 25]
# # Bastankhah
# sig0z_b = cos.(TILT.*pi/180)/sqrt(8)

# """CONSIDER NOT TRYING TO FIT TO NEGATIVE TILT AND JUST FOCUS ON THE BENEFITS OF POSITIVE TILT"""

# """TO PROVE NEGATIVE TILT IS BAD WE CAN RUN THREE TURBINE CASE WITH NEGATIVE TILT ANGLES AND LOOK AT POWER"""


# """Looking at deriving deflection"""
# ky = [-0.033, -0.021, -0.012, 0.004, 0.016, 0.032, 0.141]
# kz = [0.047, 0.04, 0.031, 0.009, -0.003, -0.007, -0.027]


# sig0y = [0.6324600220803214, 0.5420122398617231, 0.45555274142164887, 0.35337147482723696, 0.3178969525976916, 0.22800268824662198, -0.2302104789208812]
# sig0z = [0.040171620571319316, 0.06732198572068403, 0.1269824071340052, 0.3225851494794991, 0.32627652890037356, 0.29616027651514, 0.2650910334989699]
# # plot(sig0y[4:end], TILT[4:end])




x_end = 12
# # # tilt -35
# # defl_35, xvar_35 = deflection_call(1, ky, kz, sig0y, sig0z, TILT, I, CT, x_end)

# # # tilt -20
# # defl_20, xvar_20 = deflection_call(2, ky, kz, sig0y, sig0z, TILT, I, CT)

# # # tilt -15
# # defl_15, xvar_15 = deflection_call(3, ky, kz, sig0y, sig0z, TILT, I, CT)
# # TILT = [-40, -25, -20, -10, 0, 5, 20]
# TILT = [-35, -20, -15, -5, 5, 10, 25]

# # alpha = 3.0
# # beta = 0.5
# alpha = 2.32
# beta = 0.154
# I = 0.08
# CT = 0.7
# # alpha = 2.32;
#     # Beta = 0.154;
# # base tilt -5
# defl_0, xvar_0 = deflection_call(4, ky, kz, sig0y, sig0z, TILT, I, CT, alpha, beta)

# # tilt 5
# defl_5, xvar_5 = deflection_call(5, ky, kz, sig0y, sig0z, TILT, I, CT, alpha, beta)

# # tilt 10
# defl_10, xvar_10 = deflection_call(6, ky, kz, sig0y, sig0z, TILT, I, CT, alpha, beta)

# # tilt 25
# defl_25, xvar_25 = deflection_call(7, ky, kz, sig0y, sig0z, TILT, I, CT, alpha, beta)

# # # SOWFA deflection

# # X_D = [4, 5, 6, 7, 8, 9, 10, 11]
# # Z_C_0 = [99.63967017354084, 98.45427852648643, 96.59368232105771, 91.95162426180555, 85.0814652582951, 76.46572702298704, 69.9175617140443, 60.266311677827595] #Z
# # Z_C_5 = [88.4061997235887, 87.6668665653627, 83.53242730274529, 80.02195980208653, 73.8526132629067, 67.81575780501393, 63.00202558179132, 57.41052962659714] #Z
# # Z_C_10 = [83.49768448270565, 81.74066544888844, 76.83889812534574, 71.6312013311421, 67.3194618675204, 63.20050680722589, 58.25342380244782, 53.2284885542085] #Z
# # Z_C_25 = [67.25371769433993, 62.130306440546086, 58.4666343552621, 55.25496964164948, 51.66377112780846, 48.50200230182843, 44.92602406185611, 41.753554873174146] #Z


# d = D
# plot(xvar_0/d, 1 .-defl_0/90,color=:red, label="S \\gamma=-5")
# plot!(xvar_5/d, 1 .-defl_5/90, color=:blue, label="S \\gamma=5")
# plot!(xvar_10/d,1 .-defl_10/90, color=:green, label="S \\gamma=10")
# plot!(xvar_25/d, 1 .-defl_25/90, color=:black, label="S \\gamma=25")
# plot!(X_D, Z_C_0/90,seriestype=:scatter,markershape=:circle, color=:red, label="B \\gamma=-5")
# plot!(X_D, Z_C_5/90,seriestype=:scatter,markershape=:square, color=:blue, label="B \\gamma=5")
# plot!(X_D, Z_C_10/90,seriestype=:scatter,markershape=:star, color=:green, label="B \\gamma=10")
# plot!(X_D, Z_C_25/90,seriestype=:scatter,markershape=:star5, color=:black, label="B \\gamma=25", ylabel="\\delta/HH", xrotation = 0, xlabel="X/D", legend=:outertopright, grid=false)
# xlims!((0,16))
# savefig("zdeflection_new.png")


# seriestype=:scatter,markershape=:square,color=:orange
# seriestype=:scatter,markershape=:star,color=:purple
# seriestype=:scatter,markershape=:star5,color=:magenta
# seriestype=:scatter,markershape=:circle,color=:red
# seriestype=:scatter,markershape=:square,color=:blue
# seriestype=:scatter,markershape=:star,color=:green
# seriestype=:scatter,markershape=:star5,color=:black

# plot(X_D_0_y, sigmay_0,seriestype=:scatter,markershape=:circle, color=:red, label="\\gamma=-5")
# plot!(X_D_5_y, sigmay_5,seriestype=:scatter,markershape=:square, color=:blue,  label="\\gamma=5")
# plot!(X_D_10_y, sigmay_10,seriestype=:scatter,markershape=:star, color=:green, label="\\gamma=10")
# plot!(X_D_25_y, sigmay_25,seriestype=:scatter,markershape=:star5, color=:black,  label="\\gamma=25")
# plot!(X_D_0_y, Z_values_0,color=:red,label="ky=$ky_0")
# plot!(X_D_5_y, Z_values_5,color=:blue,label="ky=$ky_5")
# plot!(X_D_10_y,Z_values_10,color=:green, label="ky=$ky_10")
# plot!(X_D_25_y, Z_values_25,color=:black,label="ky=$ky_25", ylabel="\\sigma_y/D", xrotation = 0, xlabel="X/D", legend=:outertopright)



# x_0 = d*(cos(tilt*pi/180)*(1+sqrt(1-C_T)))/(sqrt(2)*((alpha*I) + (beta*(1-sqrt(1-C_T)))))
# x = x_0:3000

# sigma_z = kz.*(x.-x_0)/d .+ sigma0x
# sigma_z_g = 0.1

# # find where sigma_z is below ground threshold
# index = findall(x -> x < sigma_z_g, sigma_z)
# ind = index[1]
# x1 = x[1]:x[ind]
# x2 = x[ind]:3000


# c1 = sqrt(((sigma0x^2)/(kz^2)) - (2*sigma0x*sigma0y/(ky*kz)) + ((sigma0y^2)/ky^2))
# b1 = ((sigma0x/kz)+(sigma0y/ky)+c1)*(x1.-x_0) .+ (2*d*sigma0x*sigma0y/(ky*kz))
# a1 = ((sigma0x/kz)+(sigma0y/ky)-c1)*(x1.-x_0) .+ (2*d*sigma0x*sigma0y/(ky*kz))

# c2 = sqrt(((sigma0x^2)/(kz2^2)) - (2*sigma0x*sigma0y/(ky*kz2)) + ((sigma0y^2)/ky^2))
# b2 = ((sigma0x/kz2)+(sigma0y/ky)+c2)*(x2.-x_0) .+ (2*d*sigma0x*sigma0y/(ky*kz2))
# a2 = ((sigma0x/kz2)+(sigma0y/ky)-c2)*(x2.-x_0) .+ (2*d*sigma0x*sigma0y/(ky*kz2))


# sigmay = (ky.*(x.-x_0) .+ (d/(sqrt(8))))
# sigmaz = (-kz.*(x.-x_0) .+ ((d*cos(tilt*pi/180))/(sqrt(8))))
# a_norm = (1.6 + sqrt(C_T)).*(1.6.*sqrt.((8 .*sigmay.*sigmaz)/(d^2 * cos(tilt*pi/180))).-sqrt(C_T))
# b_norm = (1.6 - sqrt(C_T)).*(1.6.*sqrt.((8 .*sigmay.*sigmaz)/(d^2 * cos(tilt*pi/180))).+sqrt(C_T))

# inlog1 = (b1./a1)
# inlog2 = (b2./a2)
# plot(x1/d, inlog1)
# plot!(x2/d, inlog2)
# # inlog[inlog.<0.00001] .= 0
# theta_c0 = (0.3*(tilt*pi/180)/cos(tilt*pi/180))*(1-sqrt(1-C_T*cos(tilt*pi/180)))

# # Convert theta to degrees
# # theta_c0 = theta_c0 * (180/pi)
# delta_0 = theta_c0 * x_0


# e_0 = (2.93 + 1.2607*sqrt(1-C_T)) - C_T

# val1 = (sigma0x*sigma0y*theta_c0*e_0)/(1.6*ky*kz*(d^2)*c1)
# val2 = (sigma0x*sigma0y*theta_c0*e_0)/(1.6*ky*kz2*(d^2)*c2)


# val_normal = (theta_c0*d/14.7)*(sqrt(cos(tilt*pi/180)/(ky*-kz*C_T)))*e_0
# delta_normal = delta_0 .+ val_normal.*(log.(a_norm./b_norm))
# # val = -1
# d1 = val1*(log.(b1./a1))


# """Things that are wrong"""
# # log.(b1./a1) is wrong. Should be asymptoting, but instead 
# # exponentially decreases

# # delta1 = delta_0 .+val1*(log.(b1./a1))
# # delta2 = delta_0 .+val2*(log.(b2./a2)).+d1[end]
# delta1 = delta_0 .-val_normal*(log.(b1./a1))
# delta2 = delta_0 .-val_normal*(log.(b2./a2)).+d1[end]
# plot(x1/d, delta1)
# plot!(x2/d, delta2)
# plot!(x/d, delta_normal)

# val3 = -(sigma0x^2/kz^2) + (2*sigma0x*sigma0y/(ky*kz)) + -(sigma0y^2/ky^2)
# # Getting negative in sqrt even with ky=kz.
# # Need to go through derivation again, something isn't right.
#         # Maybe derivation is right, just need to solve for delta all the way

# """Validating Tilt Deflection Value"""

# # Looking at how sigz0 matches predicted
# kz = [-0.017, -0.025, -0.026, -0.039]
# ky = [0.006, 0.017, 0.02, 0.028]

# sig0y = [0.3316, 0.29196, 0.29352, 0.3321666]
# sig0z = [0.3822, 0.3747, 0.32112, 0.22124]

# tilt = [-5, 5, 10, 25]

# predicted = [1/sqrt(8), 1/sqrt(8), 1/sqrt(8), 1/sqrt(8)]
# pred_tilt = cos.(tilt.*pi/180)/sqrt(8)


# # sigma0y = 0.15441985668899924
# # sigma0x = 0.29874
# d = 126.0
# # Bastankhah numbers

# C_T = 0.8        # CT
# alpha = 2.32
# beta = 0.154
# I = 0.08
# # tilt = -15       # degrees

# x_0 = d*(cos.(tilt.*pi/180)*(1+sqrt(1-C_T)))/(sqrt(2)*((alpha*I) + (beta*(1-sqrt(1-C_T)))))
# x = x_0:3000

# sigma_z = kz.*(x.-x_0)/d .+ sigma0x
# sigma_z_g = 0.1


