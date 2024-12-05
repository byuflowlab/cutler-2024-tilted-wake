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
using Snopt
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

mutable struct params_struct{}
    Uinf
    slice
    tilt
    file1
    file2
    file3
    file4
    file5
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
            scatter!([0.0], [1.0],markerstrokewidth=0, markersize = marks, color="orange1", label="")
            # # scatter!(([Y_cm_comp_threshold].-2500)/126, [Z_cm_comp_threshold]/90, color="brown", label="Center based on max Deficit")
            # # scatter!(([Y_cm_comp_max].-2500)/126, [Z_cm_comp_max]/90, color="magenta", label="Center based on max Deficit")
            scatter!(([Y_cm_comp].-2500)/126, [Z_cm_comp]/90,markerstrokewidth=0, markersize = marks, color="green", label="Wake Center")
            plot!(x_values, y_values, label="Rotor",linewidth=linew+1,legend=:false, color="orange1", xlabel="y/D", ylabel="z*", xrotation = 0)

            xlims!((-2,2))
            ylims!((0,3.388888889))
            # plot!(size=((14/8)*400,400),yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz,foreground_color_legend = nothing,)

            plot!(size=((14/8)*400,400),yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz,foreground_color_legend = nothing)
            savefig(moviename)

        end
    end

    return sigmay, sigmaz, defz, defy, sigmaz_ground, cup, cdown, power, power_up, avg_V
end

function SOWFA_analysis_save(slice, movname, gauss_namez, gauss_namey, tol_comp, file_1, smoothing, tilt)


    avg_factor = smoothing
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
    print("SLICE: ", slice, "\n")
    slice = gridpoint((T_pos + slice*D), 10.0)
    moviename = join([movname, "_", string(slice), "_", smoothing, "_n_no_legend", ".pdf"])
    power_filename = join([movname, "_", string(slice), "_", smoothing, "_power_slice_downstream", ".pdf"])
    power_filename_down = join([movname, "_", string(slice), "_", smoothing, "_power_slice_upstream", ".pdf"])
    gauss_name_y = join([gauss_namey, "_", string(slice), "_", smoothing, "_n", ".pdf"])
    gauss_name_z = join([gauss_namez, "_", string(slice), "_", smoothing, "_n", ".pdf"])
    gauss_name_z_full = join([gauss_namez, "_", string(slice), "_", smoothing, "full", "_n", ".pdf"])
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


    fontsz = 13
    linew = 2
    marks = 5



    # sigmay[i,j] = sigma_y_d_comp
    # sigmaz[i,j] = sigma_z_d_comp
    # sigmaz_ground[i,j] = sigma_z_d_comp_ground
    # defz[i,j] = Z_cm_comp
    # defy[i,j] = Y_cm_comp
    # cup[i,j] = multiply_c
    # cdown[i,j] = multiply_c_ground
    power_downstream, avg_vel = power_estimate(cropped_SOWFA_data_comp_fine, 90.0, 2500.0, D, crop_Z_plot_fine, crop_Y_plot_fine, power_filename)
    # power_upstream = power_estimate_tilt(cropped_SOWFA_data_ambient_fine,90.0, 2500.0, D, crop_Z_plot_fine, crop_Y_plot_fine, tilt, 1.88, power_filename_down)
    # power[i,j] = power_downstream
    # power_up[i,j] = power_upstream
    # avg_V[i,j] = avg_vel


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

    return cropped_SOWFA_data_comp_neg_fine, avg_vel
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

function vel_deficit(tilt, X_D, def_z, def_y, C_T, sigy, sigz, Vel_slice, alpha_in)
    # determine where we are looking at velocity deficit (@ X_D)
    # parameters:
    # wakesave = join([string(tilt), "_", string(X_D), ".pdf"])
    
    def_z = def_z-90.0
    # print("tilt: ", tilt, "\n")
    # print("X_D: ", X_D, "\n")
    # print("def_z: ", def_z, "\n")
    # print("C_T: ", C_T, "\n")
    D = 126.0
    # sample randomly from within downstream rotor swept areas
    # create x and y values within swept area

    # import the SOWFA slice
    y_array = 2245.0:(2745.0-2245.0)/50:2745.0
    z_array = 5.0:(304.99999999994 - 5.0)/30:304.999999999994

    Y_values = y_array .- 2500.0
    Z_values = z_array .- 90.0

    # Vel_deficit_grid = zeros(length(Y_values), length(Z_values))
    # Vel_deficit_grid = []
    Total_diff = 0
    for i = 1:length(Y_values)
        # Vel_row = []
        for j = 2:length(Z_values)
            Y_val = Y_values[i]
            Z_val = Z_values[j]
            Vel_value = vel_deficit_calculations(C_T, tilt, X_D, D, Y_val, Z_val, def_z, def_y, sigy, sigz, alpha_in)
            Vel_slice_val = Vel_slice[j, i]
            # RMS
            # diff = abs((Vel_value*8.1)^3-Vel_slice_val^3)
            # Total_diff = Total_diff + diff^2

            # Power Error
            diff = abs((8.1 - Vel_value*8.1)^3-(8.1 - Vel_slice_val)^3)/((8.1 - Vel_slice_val)^3)
            Total_diff = Total_diff + diff
            # push!(Vel_row, diff)
            # push!(Vel_row, Vel_value)
        end
        # if i == 1
        #     Vel_deficit_grid = Vel_row
        # else
        #     hcat(Vel_deficit_grid, Vel_row)
        # end
    end
    # find vel_deficit at each of the points
    # Difference = sum(Vel_deficit_grid)

    # RMS
    # return sqrt(Total_diff/(length(Y_values)*length(Z_values)))

    # Power Error
    return Total_diff/(length(Y_values)*length(Z_values))
end

function vel_deficit_plot_diff(tilt, X_D, def_z, def_y, C_T, sigy, sigz, Vel_slice, alpha_in, alpha_in_0, sigz0, sigy0)
    # determine where we are looking at velocity deficit (@ X_D)
    # parameters:
    # wakesave = join([string(tilt*180/pi), "_", string(X_D), "_diff_test.pdf"])
    # wakesave1 = join([string(tilt*180/pi), "_", string(X_D), "_opt_test.pdf"])
    # wakesave2 = join([string(tilt*180/pi), "_", string(X_D), "_SOWFA_test.pdf"])
    rat = 0.62
    bw = 700
    Z_cut = 15
    default(titlefont= ("times"), guidefont=("times"), tickfont=("times"))

    wakesave = join([string(tilt*180/pi), "_", string(X_D), "_total_diff_pow_diff.pdf"])
    wakesave1 = join([string(tilt*180/pi), "_", string(X_D), "_opt_diff_pow_diff.pdf"])
    wakesave2 = join([string(tilt*180/pi), "_", string(X_D), "_not_opt_diff_pow_diff.pdf"])
    wakesave1_pow = join([string(tilt*180/pi), "_", string(X_D), "_opt_diff_pow_v3_diff.pdf"])
    wakesave2_pow = join([string(tilt*180/pi), "_", string(X_D), "_not_opt_diff_pow_v3_diff.pdf"])
    wakesave3 = join([string(tilt*180/pi), "_", string(X_D), "_SOWFA_test_initial_pow_diff.pdf"])
    wakesave4 = join([string(tilt*180/pi), "_", string(X_D), "_opt_pow_diff.pdf"])
    wakesave5 = join([string(tilt*180/pi), "_", string(X_D), "_no_opt_pow_diff.pdf"])
    # print("wakesave: ", wakesave, "\n")
    def_z = def_z-90.0
    # print("tilt: ", tilt, "\n")
    # print("X_D: ", X_D, "\n")
    # print("def_z: ", def_z, "\n")
    # print("C_T: ", C_T, "\n")
    D = 126.0
    # sample randomly from within downstream rotor swept areas
    # create x and y values within swept area

    # import the SOWFA slice
    # original SOWFA data
    y_array = 2245.0:(2745.0-2245.0)/50:2745.0
    z_array = 5.0:(304.99999999994 - 5.0)/30:304.999999999994

    # y_array = 2245.0:0.5:2745.0
    # z_array = 5.0:((304.99999999999994-5.0)/1000):304.99999999999994

    Y_values = y_array .- 2500.0
    Z_values = z_array .- 90.0
    D = 126.0

    Vel_0_grid = []
    Vel_opt_grid = []
    Vel_SOWFA_grid = []
    Vel_diff_0_grid = []
    Vel_diff_opt_grid = []
    Vel_diff_0_grid_pow = []
    Vel_diff_opt_grid_pow = []
    for i = 1:length(Y_values)
        Vel_row_opt = []
        Vel_def_row_opt = []
        Vel_def_row_no_opt = []
        Vel_SOWFA_row = []
        Vel_no_opt_row = []
        Vel_row_opt_pow = []
        Vel_no_opt_row_pow = []
        for j = 2:length(Z_values)
            Y_val = Y_values[i]

            Z_val = Z_values[j]
            Vel_value = vel_deficit_calculations(C_T, tilt, X_D, D, Y_val, Z_val, def_z, def_y, sigy, sigz, alpha_in)
            Vel_value_0 = vel_deficit_calculations(C_T, tilt, X_D, D, Y_val, Z_val, def_z, def_y, sigy0, sigz0, alpha_in_0)
            Vel_slice_val = Vel_slice[j,i]
            Vel_model = (Vel_value*8.1)
            comp = 0.4
            # diffZ = abs(Z_val)
            # # print("diffZ: ", diffZ, "\n")
            
            # diffY = abs(Y_val)
            # # print("diffY: ", diffY, "\n")
            # distance = sqrt(diffZ^2 + diffY^2)
            # # print("distance: ", distance, "\n")
            
            # if distance < D/2
            #     diff = 100*(((Vel_value*8.1))-((Vel_slice_val)))/((Vel_slice_val))
            #     diff_0 = 100*(((Vel_value_0*8.1))-((Vel_slice_val)))/((Vel_slice_val))
            # else
            #     diff = 0.0
            #     diff_0 = 0.0
            # end
            diff = 100*(((Vel_value*8.1))-((Vel_slice_val)))/((Vel_slice_val))
            diff_0 = 100*(((Vel_value_0*8.1))-((Vel_slice_val)))/((Vel_slice_val))
            # diff = 100*(((Vel_value*8.1))-((Vel_slice_val)))/((Vel_slice_val))
            # diff_0 = 100*(((Vel_value_0*8.1))-((Vel_slice_val)))/((Vel_slice_val))
            diff_pow = 100*(((8.1 - Vel_value*8.1)^3)-((8.1 - Vel_slice_val)^3))/abs((8.1 - Vel_slice_val)^3)
            diff_0_pow = 100*(((8.1 - Vel_value_0*8.1)^3)-((8.1 - Vel_slice_val)^3))/abs((8.1 - Vel_slice_val)^3)
            push!(Vel_row_opt, diff_0)
            push!(Vel_no_opt_row, diff)
            push!(Vel_row_opt_pow, diff_0_pow)
            push!(Vel_no_opt_row_pow, diff_pow)
            push!(Vel_def_row_no_opt, Vel_value)
            push!(Vel_def_row_opt, Vel_value_0)
            push!(Vel_SOWFA_row, Vel_slice_val)
            # push!(Vel_row, Vel_value)
        end
        if i == 1
            Vel_0_grid = Vel_def_row_no_opt
            Vel_opt_grid = Vel_def_row_opt
            Vel_SOWFA_grid = Vel_SOWFA_row
            Vel_diff_0_grid = Vel_no_opt_row
            Vel_diff_opt_grid = Vel_row_opt
            Vel_diff_0_grid_pow = Vel_no_opt_row_pow
            Vel_diff_opt_grid_pow = Vel_row_opt_pow

        else
            Vel_0_grid = hcat(Vel_0_grid, Vel_def_row_no_opt)
            Vel_opt_grid = hcat(Vel_opt_grid, Vel_def_row_opt)
            Vel_SOWFA_grid = hcat(Vel_SOWFA_grid, Vel_SOWFA_row)
            Vel_diff_0_grid = hcat(Vel_diff_0_grid, Vel_no_opt_row)
            Vel_diff_opt_grid = hcat(Vel_diff_opt_grid, Vel_row_opt)
            Vel_diff_0_grid_pow = hcat(Vel_diff_0_grid_pow, Vel_no_opt_row_pow)
            Vel_diff_opt_grid_pow = hcat(Vel_diff_opt_grid_pow, Vel_row_opt_pow)
        end
    end

    Vel_0_grid = convert(Matrix{Float64}, Vel_0_grid*8.1)
    Vel_opt_grid = convert(Matrix{Float64}, Vel_opt_grid*8.1)
    Vel_SOWFA_grid = convert(Matrix{Float64}, Vel_SOWFA_grid)
    Vel_diff_0_grid = convert(Matrix{Float64}, Vel_diff_0_grid)
    Vel_diff_opt_grid = convert(Matrix{Float64}, Vel_diff_opt_grid)
    Vel_diff_0_grid_pow = convert(Matrix{Float64}, Vel_diff_0_grid_pow)
    Vel_diff_opt_grid_pow = convert(Matrix{Float64}, Vel_diff_opt_grid_pow)



    Vel_total_diff = @. Vel_diff_0_grid - Vel_diff_opt_grid
    Z_values = Z_values .+ 90.0
    Z_values = convert(Array{Float64,1}, Z_values/90.0)
    Y_values = convert(Array{Float64,1}, Y_values/126.0)


    finenum = 1000
    crop_Y_plot_fine = minimum(Y_values):((maximum(Y_values)-minimum(Y_values))/finenum):maximum(Y_values)
    # print("crop_Y_plot_fine: ", crop_Y_plot_fine)
    # crop_Z_plot_fine = range(extrema(crop_Z_plot_comp)..., length=finenum)
    crop_Z_plot_fine = minimum(Z_values[2]):((maximum(Z_values)-minimum(Z_values[2]))/finenum):maximum(Z_values)
    # cropped_SOWFA_data_comp_neg_fine = [twodinterpol(x,y) for x in crop_Z_plot_fine, y in crop_Y_plot_fine]

    # Interpolate data using FLOWMath
    crop_Y_plot_fine = convert(Vector{Float64}, crop_Y_plot_fine)
    crop_Z_plot_fine = convert(Vector{Float64}, crop_Z_plot_fine)
    crop_Y_plot_comp = convert(Vector{Float64}, Y_values)
    crop_Z_plot_comp = convert(Vector{Float64}, Z_values[2:end])

    
    Vel_0_grid = interp2d(akima, crop_Z_plot_comp, crop_Y_plot_comp, Vel_0_grid, crop_Z_plot_fine, crop_Y_plot_fine)
    Vel_opt_grid = interp2d(akima, crop_Z_plot_comp, crop_Y_plot_comp, Vel_opt_grid, crop_Z_plot_fine, crop_Y_plot_fine)
    Vel_SOWFA_grid = interp2d(akima, crop_Z_plot_comp, crop_Y_plot_comp, Vel_SOWFA_grid, crop_Z_plot_fine, crop_Y_plot_fine)
    Vel_diff_0_grid = interp2d(akima, crop_Z_plot_comp, crop_Y_plot_comp, Vel_diff_0_grid, crop_Z_plot_fine, crop_Y_plot_fine)
    Vel_diff_opt_grid = interp2d(akima, crop_Z_plot_comp, crop_Y_plot_comp, Vel_diff_opt_grid, crop_Z_plot_fine, crop_Y_plot_fine)
    Vel_diff_0_grid_pow = interp2d(akima, crop_Z_plot_comp, crop_Y_plot_comp, Vel_diff_0_grid_pow, crop_Z_plot_fine, crop_Y_plot_fine)
    Vel_diff_opt_grid_pow = interp2d(akima, crop_Z_plot_comp, crop_Y_plot_comp, Vel_diff_opt_grid_pow, crop_Z_plot_fine, crop_Y_plot_fine)


    # cut out stuff outside of rotor swept area:
    for i = 1:length(crop_Y_plot_fine)
        for j = 1:length(crop_Z_plot_fine)
            Y_val = crop_Y_plot_fine[i]
            Z_val = crop_Z_plot_fine[j]
            distance = sqrt(Y_val^2 + Z_val^2)
            # print("distance: ", distance, "\n")
            Vel_value = Vel_0_grid[i,j]
            Vel_slice_val = Vel_SOWFA_grid[i,j]
            if (Vel_value < 1.3) && (Vel_slice_val < 1.3)
                Vel_diff_0_grid[i,j] = 0.0
                Vel_diff_opt_grid[i,j] = 0.0
            end
        end
    end

    # print("length(Z_values): ", length(Z_values), "\n")
    # print("length(Y_values): ", length(Y_values), "\n")
    # print("size(Vel_0_grid): ", size(Vel_0_grid), "\n")

    f(t) = ((126/2).*cos.(t))/126
    g(t) = (90 .+ (126/2).*sin.(t))/90
    range = 0:0.0001:2*pi
    x_values = f(range)
    y_values = g(range)

    # # scatter!((crop_Y_plot_comp[Y_whole_wake_comp].-2500)/126, (crop_Z_plot_comp[X_whole_wake_comp])/90, color="red", label="Wake")
    

    fontsz = 13
    linew = 2
    marks = 20
    ms = 3

    fontylabel = text("", 15).font
    fontylabel.rotation=90

    fontxlabel = text("", 15).font
    fontxlabel.rotation=0
    # find vel_deficit at each of the points
    
    
    data1 = heatmap(crop_Y_plot_fine, crop_Z_plot_fine, Vel_0_grid, c=:Blues, clim=(0.0, 2.5))
    plot(data1, colorbar_title=" \nVelocity Deficit (m/s)",  xlimit = [-1.0, 1.0],bottom_margin=10Plots.mm, right_margin=5Plots.mm, grid=:false, xlabel="y/D",yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz)
    scatter!([0.0], [1.0],linewidth = ms,markerstrokewidth=ms, markershape=:+, markersize = marks, color="orange1", label="")
    plot!(x_values, y_values, label="Rotor",linewidth=linew+1,legend=:false, color="orange1", xlabel="y/D",ylabel=" ", xrotation = 0)
    plot!(size=(bw,bw*rat))
    savefig(wakesave5)

    data1 = heatmap(crop_Y_plot_fine, crop_Z_plot_fine, Vel_opt_grid, c=:Blues, clim=(0.0, 2.5))
    plot(data1, colorbar_title=" \nVelocity Deficit (m/s)", ylimit = [crop_Z_plot_fine[Z_cut], 2.0], xlimit = [-1.0, 1.0],bottom_margin=10Plots.mm, right_margin=5Plots.mm, grid=:false, xlabel="y/D",ylabel=" ",yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz)
    scatter!([0.0], [1.0],linewidth = ms,markerstrokewidth=ms, markershape=:+, markersize = marks, color="orange1", label="")
    plot!(x_values, y_values, label="Rotor",linewidth=linew+1,legend=:false, color="orange1", xlabel="y/D",ylabel=" ", xrotation = 0)
    plot!(size=(bw,bw*rat))
    savefig(wakesave4)

    data1 = heatmap(crop_Y_plot_fine, crop_Z_plot_fine, Vel_SOWFA_grid, c=:Blues, clim=(0.0, 2.5))
    plot(data1, colorbar_title=" \nVelocity Deficit (m/s)", ylimit = [crop_Z_plot_fine[Z_cut], 2.0], xlimit = [-1.0, 1.0],bottom_margin=10Plots.mm, right_margin=5Plots.mm, grid=:false, xlabel="y/D",ylabel=" ",yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz)
    scatter!([0.0], [1.0],linewidth = ms,markerstrokewidth=ms, markershape=:+, markersize = marks, color="orange1", label="")
    plot!(x_values, y_values, label="Rotor",linewidth=linew+1,legend=:false, color="orange1", xlabel="y/D",ylabel=" ", xrotation = 0)
    plot!(size=(bw,bw*rat))
    savefig(wakesave3)

    data1 = heatmap(crop_Y_plot_fine, crop_Z_plot_fine, Vel_diff_0_grid, c=:RdBu, clim=(-20, 20))
    plot(data1, xlimit = [-1.0, 1.0], colorbar_title="",bottom_margin=10Plots.mm,  right_margin=15Plots.mm,ylimit = [crop_Z_plot_fine[Z_cut], 2.0], grid=:false, xlabel="y/D",ylabel=" ",yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz)
    scatter!([0.0], [1.0],linewidth = ms,markerstrokewidth=ms, markershape=:+, markersize = marks, color="orange1", label="")
    plot!(x_values, y_values, label="Rotor",linewidth=linew+1,legend=:false, color="orange1", xlabel="y/D",ylabel=" ", xrotation = 0)
    plot!(annotations = ([1.5], 1.1,text("Relative Error       (%)", fontylabel)))
    plot!(annotations = ([1.505], 1.4,text(L"\Delta U", fontylabel)))
    plot!(annotations = ([-1.28], 1.1,text("z*", fontxlabel)))
    plot!(size=(bw,bw*rat))
    savefig(wakesave2)

    data1 = heatmap(crop_Y_plot_fine, crop_Z_plot_fine, Vel_diff_opt_grid, c=:RdBu, clim=(-20, 20))
    plot(data1, xlimit = [-1.0, 1.0], colorbar_title="",bottom_margin=10Plots.mm,  right_margin=15Plots.mm,ylimit = [crop_Z_plot_fine[Z_cut], 2.0], grid=:false, xlabel="y/D",ylabel=" ",yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz)
    scatter!([0.0], [1.0],linewidth = ms,markerstrokewidth=ms, markershape=:+, markersize = marks, color="orange1", label="")
    plot!(x_values, y_values, label="Rotor",linewidth=linew+1,legend=:false, color="orange1", xlabel="y/D",ylabel=" ", xrotation = 0)
    plot!(annotations = ([1.5], 1.1,text("Relative Error       (%)", fontylabel)))
    plot!(annotations = ([1.505], 1.4,text(L"\Delta U", fontylabel)))
    plot!(annotations = ([-1.28], 1.1,text("z*", fontxlabel)))
    plot!(size=(bw,bw*rat))
    savefig(wakesave1)

    data1 = heatmap(crop_Y_plot_fine, crop_Z_plot_fine, Vel_diff_0_grid_pow, c=:RdBu, clim=(-15, 15))
    plot(data1, xlimit = [-1.0, 1.0], colorbar_title="",bottom_margin=10Plots.mm,  right_margin=15Plots.mm,ylimit = [crop_Z_plot_fine[Z_cut], 2.0], grid=:false, xlabel="y/D",ylabel=" ",yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz)
    scatter!([0.0], [1.0],linewidth = ms,markerstrokewidth=ms, markershape=:+, markersize = marks, color="orange1", label="")
    plot!(x_values, y_values, label="Rotor",linewidth=linew+1,legend=:false, color="orange1", xlabel="y/D",ylabel=" ", xrotation = 0)
    plot!(annotations = ([1.5], 1.1,text("Relative Error       (%)", fontylabel)))
    plot!(annotations = ([1.505], 1.4,text(L"U^3", fontylabel)))
    plot!(annotations = ([-1.28], 1.1,text("z*", fontxlabel)))
    plot!(size=(bw,bw*rat))
    savefig(wakesave2_pow)

    data1 = heatmap(crop_Y_plot_fine, crop_Z_plot_fine, Vel_diff_opt_grid_pow, c=:RdBu, clim=(-15, 15))
    plot(data1, xlimit = [-1.0, 1.0], colorbar_title="",bottom_margin=10Plots.mm,  right_margin=15Plots.mm,ylimit = [crop_Z_plot_fine[Z_cut], 2.0], grid=:false, xlabel="y/D",ylabel=" ",yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz)
    scatter!([0.0], [1.0],linewidth = ms,markerstrokewidth=ms, markershape=:+, markersize = marks, color="orange1", label="")
    plot!(x_values, y_values, label="Rotor",linewidth=linew+1,legend=:false, color="orange1", xlabel="y/D",ylabel=" ", xrotation = 0)
    plot!(annotations = ([1.5], 1.1,text("Relative Error       (%)", fontylabel)))
    plot!(annotations = ([1.505], 1.4,text(L"U^3", fontylabel)))
    plot!(annotations = ([-1.28], 1.1,text("z*", fontxlabel)))
    plot!(size=(bw,bw*rat))
    savefig(wakesave1_pow)

    # RdBu
    # data1 = heatmap(crop_Y_plot_fine, crop_Z_plot_fine, Vel_total_diff, c=:RdBu, clim=(-.2, .2))
    # plot(data1, colorbar_title=" \\n \\n |\\Delta V_{orig}| - |\\Delta V_{opt}|", xlimit = [-1.0, 1.0], right_margin=15Plots.mm, grid=:false, xlabel="y/D",ylabel=" ",yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz)
    # savefig(wakesave)
    
    # data1 = heatmap(crop_Y_plot_fine, crop_Z_plot_fine, Vel_total_diff, c=:RdBu, clim=(-5, 5))
    # plot(data1, xlimit = [-1.0, 1.0],bottom_margin=10Plots.mm, right_margin=15Plots.mm, grid=:false, xlabel="y/D",ylabel=" ",yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz)
    # plot!(annotations = ([1.28], 1.1,text(L"\left|\Delta V_{org}\right| - |\Delta V_{opt}|", fontylabel)))
    # savefig(wakesave)

    # wakesave = join([string(tilt*180/pi), "_", string(X_D), "_total_diff.pdf"])
    # wakesave1 = join([string(tilt*180/pi), "_", string(X_D), "_opt_diff.pdf"])
    # wakesave2 = join([string(tilt*180/pi), "_", string(X_D), "_not_opt_diff.pdf"])
    # wakesave3 = join([string(tilt*180/pi), "_", string(X_D), "_SOWFA_test_initial.pdf"])
    # wakesave4 = join([string(tilt*180/pi), "_", string(X_D), "_opt.pdf"])
    # wakesave5 = join([string(tilt*180/pi), "_", string(X_D), "_no_opt.pdf"])
end

function vel_deficit_plot(tilt, X_D, def_z, def_y, C_T, sigy, sigz, Vel_slice, alpha_in)
    # determine where we are looking at velocity deficit (@ X_D)
    # parameters:
    # wakesave = join([string(tilt*180/pi), "_", string(X_D), "_diff_test.pdf"])
    # wakesave1 = join([string(tilt*180/pi), "_", string(X_D), "_opt_test.pdf"])
    # wakesave2 = join([string(tilt*180/pi), "_", string(X_D), "_SOWFA_test.pdf"])

    wakesave = join([string(tilt*180/pi), "_", string(X_D), "_diff_test_initial.pdf"])
    wakesave1 = join([string(tilt*180/pi), "_", string(X_D), "_opt_test_intial.pdf"])
    wakesave2 = join([string(tilt*180/pi), "_", string(X_D), "_SOWFA_test_initial.pdf"])
    # print("wakesave: ", wakesave, "\n")
    def_z = def_z-90.0
    # print("tilt: ", tilt, "\n")
    # print("X_D: ", X_D, "\n")
    # print("def_z: ", def_z, "\n")
    # print("C_T: ", C_T, "\n")
    D = 126.0
    # sample randomly from within downstream rotor swept areas
    # create x and y values within swept area

    # import the SOWFA slice
    # original SOWFA data
    y_array = 2245.0:(2745.0-2245.0)/50:2745.0
    z_array = 5.0:(304.99999999994 - 5.0)/30:304.999999999994

    # y_array = 2245.0:0.5:2745.0
    # z_array = 5.0:((304.99999999999994-5.0)/1000):304.99999999999994

    Y_values = y_array .- 2500.0
    Z_values = z_array .- 90.0

    Vel_deficit_grid = []
    Vel_opt_grid = []
    Vel_SOWFA_grid = []
    for i = 1:length(Y_values)
        Vel_row = []
        Vel_def_row = []
        Vel_SOWFA_row = []
        for j = 2:length(Z_values)
            Y_val = Y_values[i]

            Z_val = Z_values[j]
            Vel_value = vel_deficit_calculations(C_T, tilt, X_D, D, Y_val, Z_val, def_z, def_y, sigy, sigz, alpha_in)
            Vel_slice_val = Vel_slice[j,i]
            diff = abs((Vel_value*8.1)^3-Vel_slice_val^3)
            push!(Vel_row, diff)
            push!(Vel_def_row, Vel_value)
            push!(Vel_SOWFA_row, Vel_slice_val)
            # push!(Vel_row, Vel_value)
        end
        if i == 1
            Vel_deficit_grid = Vel_row
            Vel_opt_grid = Vel_def_row
            Vel_SOWFA_grid = Vel_SOWFA_row

        else
            Vel_deficit_grid = hcat(Vel_deficit_grid, Vel_row)
            Vel_opt_grid = hcat(Vel_opt_grid , Vel_def_row)
            Vel_SOWFA_grid = hcat(Vel_SOWFA_grid, Vel_SOWFA_row)
        end
    end

    Vel_deficit_grid = convert(Matrix{Float64}, Vel_deficit_grid)
    Vel_opt_grid = convert(Matrix{Float64}, Vel_opt_grid*8.1)
    Vel_SOWFA_grid = convert(Matrix{Float64}, Vel_SOWFA_grid)
    # find vel_deficit at each of the points
    data1 = heatmap(Vel_deficit_grid, c=:Blues)
    plot(data1, colorbar_title=" \nVelocity Deficit (m/s)", right_margin=5Plots.mm, grid=:false, xlabel="y/D", ylabel="z*")
    savefig(wakesave)

    data1 = heatmap(Vel_opt_grid, c=:Blues)
    plot(data1, colorbar_title=" \nVelocity Deficit (m/s)", right_margin=5Plots.mm, grid=:false, xlabel="y/D", ylabel="z*")
    savefig(wakesave1)

    data1 = heatmap(Vel_SOWFA_grid, c=:Blues)
    plot(data1, colorbar_title=" \nVelocity Deficit (m/s)", right_margin=5Plots.mm, grid=:false, xlabel="y/D", ylabel="z*")
    savefig(wakesave2)
end


# function vel_deficit_calculations(C_T, tilt, X_D, D, y_val, z_val, deflection, sigy_D, sigz_D)
#     # sigy_D, sigz_D = sigysigz(X_D, tilt)
#     sigy = sigy_D*D
#     sigz = sigz_D*D
#     # print("sigy: ", sigy, "\n")
#     # print("sigz: ", sigz, "\n")
#     b = (C_T*cos(tilt))/(8*(sigy_D*sigz_D))
#     c = exp(-0.5*(y_val/sigy)^2)
#     d = exp(-0.5*((z_val-deflection)/sigz)^2)

#     if (1-b) < 0.0
#         vel_deficit_value = 100
#     else
#         vel_deficit_value = (1 - sqrt(1 - b))*c*d
#     end
#     return vel_deficit_value
# end

function vel_deficit_calculations(C_T, tilt, X_D, D, y_val, z_val, deflection, deflection_y, sigy_D, sigz_D, alpha_in)
    # sigy_D, sigz_D = sigysigz(X_D, tilt)
    sigy = sigy_D*D
    sigz = sigz_D*D
    # print("sigy: ", sigy, "\n")
    # print("sigz: ", sigz, "\n")
    
    # is this X/D or just X?
    y_add = X_D*D*tan(alpha_in)
    b = (C_T*cos(tilt))/(8*(sigy_D*sigz_D))
    c = exp(-0.5*((y_val+y_add + deflection_y)/sigy)^2)
    d = exp(-0.5*((z_val-deflection)/sigz)^2)

    vel_deficit_value = (1 - sqrt(1 - b))*c*d
    return vel_deficit_value
end

function SOWFA_multiple_save(movnames, files, filenames, tilt)
    for j in 1:length(files)
        file_1 = files[j]
        filename = filenames[j]
        movnam = movnames[j]

        # slice = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        D = 126
        # slice = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        slices = 7:1.0:12
        z_array = 1001
        y_array = 1001
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

        # d7 = zeros(length(slices))
        # d8 = zeros(length(slices), length(smoothing))
        # d9 = zeros(length(slices), length(smoothing))
        # d10 = zeros(length(slices), length(smoothing))
        # d11 = zeros(length(slices), length(smoothing))
        # d12 = zeros(length(slices), length(smoothing))
        avgV_val = zeros(length(slices))

        Slices_vel = zeros(z_array, y_array, length(slices))



        moviename = join([movnam])
        gauss_name_z = join([movnam, "_gauss_z"])
        gauss_name_y = join([movnam, "_gauss_y_new"])

        print(moviename)
        for i in 1:length(slices)
            print("slice: ", slices[i], "\n")
            print("slices: ", slices, "\n")
            print("i: ", i, "\n")
            Slices_vel[:,:,i], avgV_val[i] = SOWFA_analysis_save(slices[i], moviename, gauss_name_z, gauss_name_y, threshold, file_1, smoothing, tilt[j])
        end


        save(filename, "Vel_slices", Slices_vel, "avgV", avgV_val)
    end
end

function SOWFA_multiple(movnames, files, filenames, tilt)
    for i in 1:length(files)
        file_1 = files[i]
        filename = filenames[i]
        movnam = movnames[i]

        # slice = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        D = 126
        # slice = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        slices = 6:0.2:15
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

function deflection_est_y(tilt, x_D)
    c1 = -1.7557859060914274
    c2 = 2.432276709550188
    c3 = -0.012450608058747908
    c4 = 0.31866455652218834
    c5 = -0.31314460132041544
    c6 = -0.008109189965536603
    c7 = 0.021240407783873518

    defz = @. (c1*tilt) + (c2*tilt^2) + (c3*x_D) + (c4*x_D*tilt) + (c5*(tilt^2)*x_D) + (c6*(x_D^2)*tilt) + c7
    return defz
end

function objective_vel_wrapper!(g, x, params)
    # parameters
    U_inf = params.Uinf
    slices = params.slice
    tilt_angles = params.tilt
    f1 = params.file1
    f2 = params.file2
    f3 = params.file3
    f4 = params.file4
    f5 = params.file5

    # load trainable parameters
    # for CT model
    CTA = x[1]
    CTB = x[2]
    CTC = x[3]
    CTη = x[4]
    CTb = x[5]
    # for kz and ky
    kz1 = x[6]
    kz2 = x[7]
    kz3 = x[8]
    ky1 = x[9]
    ky2 = x[10]
    # for sigy0 and sigz0
    sigy01 = x[11]
    sigz01 = x[12]
    sigz02 = x[13]
    sigz03 = x[14]
    # alpha for veer
    alpha_in = x[15]

    

    CT_val = @. CTA*Gompertz_dist((tilt_angles*CTC).+CTB, CTη, CTb)
    CT_val = @. CT_val/CT_val
    # CT_val = CT_val*0.7620929940139257
    CT_val = CT_val*0.83
    filenames = [f1, f2, f3, f4, f5]
    objective_total = 0.0

    # solve for constraints
    constraint = one_minus_b(x, slices, CT_val, params)
    g[:] = constraint

    for i = 1:length(tilt_angles)
        tilt_val = tilt_angles[i]
        filename = filenames[i]
        # find delta V based on the model for all slices and tilt angles
        delta_V_model = delta_V_pertilt(tilt_val, slices, CT_val[i], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03, filename, alpha_in)
        # delta_V_SOWFA = (load(filename, "avgV").-U_inf)/U_inf     # avgV @ tilt[i] for X_D = 7.0:1.0:12.0
        # objective = @. abs(delta_V_SOWFA - delta_V_model)
        objective = sum(delta_V_model)/length(slices)
        objective_total = objective_total + objective
    end

    # normalize objective with respect to tilt angles
    return objective_total*100/length(tilt_angles)
end

function objective_vel_wrapper_snopt!(g, df, dg, x, deriv, params)
    # parameters
    U_inf = params.Uinf
    slices = params.slice
    tilt_angles = params.tilt
    f1 = params.file1
    f2 = params.file2
    f3 = params.file3
    f4 = params.file4
    f5 = params.file5

    # load trainable parameters
    # for CT model
    CTA = x[1]
    CTB = x[2]
    CTC = x[3]
    CTη = x[4]
    CTb = x[5]
    # for kz and ky
    kz1 = x[6]
    kz2 = x[7]
    kz3 = x[8]
    ky1 = x[9]
    ky2 = x[10]
    # for sigy0 and sigz0
    sigy01 = x[11]
    sigz01 = x[12]
    sigz02 = x[13]
    sigz03 = x[14]
    # alpha for veer
    alpha_in = x[15]

    

    CT_val = @. CTA*Gompertz_dist((tilt_angles*CTC).+CTB, CTη, CTb)
    CT_val = @. CT_val/CT_val
    CT_val = CT_val*0.83
    # CT_val = CT_val*0.7620929940139257
    filenames = [f1, f2, f3, f4, f5]
    objective_total = 0.0

    # solve for constraints
    constraint = one_minus_b(x, slices, CT_val, params)
    g[:] = constraint

    for i = 1:length(tilt_angles)
        tilt_val = tilt_angles[i]
        filename = filenames[i]
        # find delta V based on the model for all slices and tilt angles

        # this returns an array of the abs difference in Velocity between prediction and SOWFA for each slice
        delta_V_model = delta_V_pertilt(tilt_val, slices, CT_val[i], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03, filename, alpha_in)
        # delta_V_SOWFA = (load(filename, "avgV").-U_inf)/U_inf     # avgV @ tilt[i] for X_D = 7.0:1.0:12.0
        # objective = @. abs(delta_V_SOWFA - delta_V_model)
        # normalizing objective by slices and tilt angles
        objective = sum(delta_V_model)/length(slices)
        objective_total = objective_total + objective
    end

    # normalize objective with respect to tilt angles
    return objective_total/length(tilt_angles)
end

function objective_vel_wrapper_plot_difference(x, x0_opt, params)
    # parameters
    slices = 7.0:1.0:12.0
    tilt_angles = params.tilt
    # filename_opt = "objective_best_opt.jld"
    filename_opt = "objective_best_opt_x0.jld"

    f1 = params.file1
    f2 = params.file2
    f3 = params.file3
    f4 = params.file4
    f5 = params.file5

    filenames = [f1, f2, f3, f4, f5]

    # load trainable parameters
    # for CT model
    CTA = x[1]
    CTB = x[2]
    CTC = x[3]
    CTη = x[4]
    CTb = x[5]
    # for kz and ky
    kz1 = x[6]
    kz2 = x[7]
    kz3 = x[8]
    ky1 = x[9]
    ky2 = x[10]
    # for sigy0 and sigz0
    sigy01 = x[11]
    sigz01 = x[12]
    sigz02 = x[13]
    sigz03 = x[14]
    # alpha for veer
    alpha_in = x[15]

    # load trainable parameters optimized
    # for CT model
    CTA_0 = x0_opt[1]
    CTB_0 = x0_opt[2]
    CTC_0 = x0_opt[3]
    CTη_0 = x0_opt[4]
    CTb_0 = x0_opt[5]
    # for kz and ky
    kz1_0 = x0_opt[6]
    kz2_0 = x0_opt[7]
    kz3_0 = x0_opt[8]
    ky1_0 = x0_opt[9]
    ky2_0 = x0_opt[10]
    # for sigy0 and sigz0
    sigy01_0 = x0_opt[11]
    sigz01_0 = x0_opt[12]
    sigz02_0 = x0_opt[13]
    sigz03_0 = x0_opt[14]
    # alpha for veer
    alpha_in_0 = x0_opt[15]

    

    CT_val = @. CTA*Gompertz_dist((tilt_angles*CTC).+CTB, CTη, CTb)
    CT_val = @. CT_val/CT_val
    # CT_val = CT_val*0.83
    CT_val = CT_val*0.7620929940139257
    U_inf = 8.1
    # solve for constraints
    # constraint = one_minus_b(x, slices, CT_val, params)
    # g[:] = constraint
    save_best_opt = []
    for i = 1:length(tilt_angles)
        tilt_val = tilt_angles[i]
        filename = filenames[i]
        # find delta V based on the model for all slices and tilt angles
        delta_V_pertilt_plot_diff(tilt_val, slices, CT_val[i], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03, filename, alpha_in, alpha_in_0,
                                    kz1_0, kz2_0, kz3_0, ky1_0, ky2_0, sigy01_0, sigz01_0, sigz02_0, sigz03_0)

        # # save data for analyzing
        # delta_V_model = delta_V_pertilt(tilt_val, slices, CT_val[i], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03, filename, alpha_in)
        # # divide by number of points?
        # push!(save_best_opt, delta_V_model)
    end
    # save(filename_opt, "f_detail", save_best_opt)
end

function objective_vel_wrapper_plot(x, params)
    # parameters
    slices = 7.0:1.0:12.0
    tilt_angles = params.tilt
    filename_opt = "objective_best_opt.jld"
    # filename_opt = "objective_best_opt_x0.jld"

    f1 = params.file1
    f2 = params.file2
    f3 = params.file3
    f4 = params.file4
    f5 = params.file5

    filenames = [f1, f2, f3, f4, f5]

    # load trainable parameters
    # for CT model
    CTA = x[1]
    CTB = x[2]
    CTC = x[3]
    CTη = x[4]
    CTb = x[5]
    # for kz and ky
    kz1 = x[6]
    kz2 = x[7]
    kz3 = x[8]
    ky1 = x[9]
    ky2 = x[10]
    # for sigy0 and sigz0
    sigy01 = x[11]
    sigz01 = x[12]
    sigz02 = x[13]
    sigz03 = x[14]
    # alpha for veer
    alpha_in = x[15]

    

    CT_val = @. CTA*Gompertz_dist((tilt_angles*CTC).+CTB, CTη, CTb)
    CT_val = @. CT_val/CT_val
    # CT_val = CT_val*0.83
    CT_val = CT_val*0.7620929940139257
    U_inf = 8.1
    # solve for constraints
    # constraint = one_minus_b(x, slices, CT_val, params)
    # g[:] = constraint
    save_best_opt = []
    for i = 1:length(tilt_angles)
        tilt_val = tilt_angles[i]
        filename = filenames[i]
        # find delta V based on the model for all slices and tilt angles
        delta_V_pertilt_plot(tilt_val, slices, CT_val[i], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03, filename, alpha_in)

        # save data for analyzing
        delta_V_model = delta_V_pertilt(tilt_val, slices, CT_val[i], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03, filename, alpha_in)
        # divide by number of points?
        # objective = sum(delta_V_model)/length(slices)
        # objective_total = objective_total + objective
        push!(save_best_opt, delta_V_model)
    end
    # print("save_best_opt: ", save_best_opt, "\n")
    save(filename_opt, "f_detail", save_best_opt)
end


function one_minus_b(x, slices, CT_val, params)
    # for kz and ky
    kz1 = x[6]
    kz2 = x[7]
    kz3 = x[8]
    ky1 = x[9]
    ky2 = x[10]
    # for sigy0 and sigz0
    sigy01 = x[11]
    sigz01 = x[12]
    sigz02 = x[13]
    sigz03 = x[14]

    tilt = params.tilt

    g = []
    sigz_all = []
    sigy_all = []
    for i = 1:length(tilt)
        for slice in slices
            tilt_val = tilt[i]
            sigy, sigz = sigysigz(slice, tilt_val, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
            b = @. (CT_val[i]*cos(tilt_val))/(8*(sigy*sigz))
            # print("CT_val[i]: ", CT_val[i], "\n")
            g_val = b-1
            push!(g, g_val)
            push!(sigz_all, sigz)
            push!(sigy_all, sigy)
        end
    end

    # for i = 1:length(CT_val)
    #     push!(g, -CT_val[i])
    # end

    for i = 1:length(CT_val)
        push!(g, -0.8)
    end

    for i = 1:length(sigz_all)
        push!(g, sigz_all[i])
        push!(g, sigy_all[i])
    end
    
    # print("g: ", g, "\n")
    return g
end

function delta_V_pertilt(tilt, slices, Ct, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03, filename, alpha_in)
    dV = []
    i = 1
    Vel_slice = load(filename, "Vel_slices")
    for slice in slices
        def_z = 90.0.+deflection_est(tilt, slice)*126.0
        def_y = deflection_est_y(tilt, slice)*126.0
        sigy, sigz = sigysigz(slice, tilt, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
        Vel_val = vel_deficit(tilt, slice, def_z, def_y, Ct, sigy, sigz, Vel_slice[:,:,i], alpha_in)
        # Vel_val = Vel_val/(31*51)
        push!(dV, Vel_val)
        i = i+1
    end
    return dV
end

function delta_V_pertilt_plot(tilt, slices, Ct, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03, filename, alpha_in)
    i = 1
    Vel_slice = load(filename, "Vel_slices")
    for slice in slices
        def_z = 90.0.+deflection_est(tilt, slice)*126.0
        def_y = deflection_est_y(tilt, slice)*126.0
        sigy, sigz = sigysigz(slice, tilt, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
        vel_deficit_plot(tilt, slice, def_z, def_y, Ct, sigy, sigz, Vel_slice[:,:,i], alpha_in)
        i = i+1
    end
end

function delta_V_pertilt_plot_diff(tilt, slices, Ct, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03, filename, alpha_in, alpha_in_0,
                                    kz1_0, kz2_0, kz3_0, ky1_0, ky2_0, sigy01_0, sigz01_0, sigz02_0, sigz03_0)
    i = 1
    Vel_slice = load(filename, "Vel_slices")
    for slice in slices
        def_z = 90.0.+deflection_est(tilt, slice)*126.0
        def_y = deflection_est_y(tilt, slice)*126.0
        sigy, sigz = sigysigz(slice, tilt, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
        sigy0, sigz0 = sigysigz(slice, tilt, kz1_0, kz2_0, kz3_0, ky1_0, ky2_0, sigy01_0, sigz01_0, sigz02_0, sigz03_0)
        vel_deficit_plot_diff(tilt, slice, def_z, def_y, Ct, sigy, sigz, Vel_slice[:,:,i], alpha_in, alpha_in_0, sigz0, sigy0)
        i = i+1
    end
end

function sigysigz(X_D, tilt, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
    kz = kz1*tilt^2 + kz2*tilt + kz3
    ky = ky1*tilt + ky2
    sigzo = sigz01 + sigz02*log(tilt+ sigz03)
    # sigzo = 0.20
    sigyo = sigy01
    # sigyo = 0.266

    sigy = (ky*X_D) + sigyo
    sigz = (kz*X_D) + sigzo
    return sigy, sigz
end

function sigysigz_solve(X_D, tilt, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
    kz = kz1*tilt^2 + kz2*tilt + kz3
    ky = ky1*tilt + ky2
    sigzo = sigz01 + sigz02*log(tilt+ sigz03)
    # sigzo = 0.20
    sigyo = sigy01
    # sigyo = 0.266

    sigy = @. (ky*X_D) + sigyo
    sigz = @. (kz*X_D) + sigzo
    return sigy, sigz
end

function sigysigz_solve(X_D, tilt, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
    kz = kz1*tilt^2 + kz2*tilt + kz3
    ky = ky1*tilt + ky2
    sigzo = sigz01 + sigz02*log(tilt+ sigz03)
    # sigzo = 0.20
    sigyo = sigy01
    # sigyo = 0.266

    sigy = @. (ky*X_D) + sigyo
    sigz = @. (kz*X_D) + sigzo
    return sigy, sigz
end

function kykzsolve(X_D, tilt, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
    kz = kz1*tilt^2 + kz2*tilt + kz3
    ky = ky1*tilt + ky2
    return ky, kz
end

function run_optimization_SNOPT()
    file_0 = "/Users/jamescutler/Downloads/byu_tilt_runs_single/c_000_sp7_1turb_hNormal_D126_tilt_base/lite_data_000.ftr"
    file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_001_sp7_1turb_hNormal_D126_tilt_2.5/lite_data_001.ftr"
    file_2 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_004_sp7_1turb_hNormal_D126_tilt_5/lite_data_004.ftr"
    file_3 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_002_sp7_1turb_hNormal_D126_tilt_7.5/lite_data_002.ftr"
    file_4 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_005_sp7_1turb_hNormal_D126_tilt_10/lite_data_005.ftr"
    file_5 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_003_sp7_1turb_hNormal_D126_tilt_12.5/lite_data_003.ftr"

    # for power gains
    # filename0 = "analysis_smoothing_piece_wise_largeest_span_base_normal_n.jld"
    # filename1 = "analysis_smoothing_piece_wise_largeest_span_2.5_normal_n.jld"
    # filename2 = "analysis_smoothing_piece_wise_largeest_span_5_normal_n.jld"
    # filename3 = "analysis_smoothing_piece_wise_largeest_span_7.5_normal_n.jld"
    # filename4 = "analysis_smoothing_piece_wise_largeest_span_10_normal_n.jld"
    # filename5 = "analysis_smoothing_piece_wise_largeest_span_12.5_normal_n.jld"

    filename0 = "analysis_smoothing_piece_wise_largeest_span_base_normal_originalSOWFA.jld"
    filename1 = "analysis_smoothing_piece_wise_largeest_span_2.5_normal_originalSOWFA.jld"
    filename2 = "analysis_smoothing_piece_wise_largeest_span_5_normal_originalSOWFA.jld"
    filename3 = "analysis_smoothing_piece_wise_largeest_span_7.5_normal_originalSOWFA.jld"
    filename4 = "analysis_smoothing_piece_wise_largeest_span_10_normal_originalSOWFA.jld"
    filename5 = "analysis_smoothing_piece_wise_largeest_span_12.5_normal_originalSOWFA.jld"
    

    # filename0 = "analysis_smoothing_piece_wise_largeest_span_base_normal_n_finall.jld"
    # filename1 = "analysis_smoothing_piece_wise_largeest_span_2.5_normal_n_finall.jld"
    # filename2 = "analysis_smoothing_piece_wise_largeest_span_5_normal_n_finall.jld"
    # filename3 = "analysis_smoothing_piece_wise_largeest_span_7.5_normal_n_finall.jld"
    # filename4 = "analysis_smoothing_piece_wise_largeest_span_10_normal_n_finall.jld"
    # filename5 = "analysis_smoothing_piece_wise_largeest_span_12.5_normal_n_finall.jld"


    Gompertz_dist(x, η, b) = @. (b*η)*exp(η + (b*x) - (η*exp(10.0*b*x)))
    # y and z array
    y_array = 2245.0:0.5:2745.0
    z_array = 5.0:((304.99999999999994-5.0)/1000):304.99999999999994

    # Final data values
    # filename0 = "analysis_smoothing_piece_wise_largeest_span_base_normal_n_final.jld"
    # filename1 = "analysis_smoothing_piece_wise_largeest_span_2.5_normal_n_final.jld"
    # filename2 = "analysis_smoothing_piece_wise_largeest_span_5_normal_n_final.jld"
    # filename3 = "analysis_smoothing_piece_wise_largeest_span_7.5_normal_n_final.jld"
    # filename4 = "analysis_smoothing_piece_wise_largeest_span_10_normal_n_final.jld"
    # filename5 = "analysis_smoothing_piece_wise_largeest_span_12.5_normal_n_final.jld"


    # filename1 = "analysis_25.jld"
    # filename2 = "analysis_neg15.jld"
    # filename1 = "analysis_smoothing_piece_wise_largeest_span_25_normal.jld"


    # files = [file_1]
    # movnames = ["25"]
    # filenames = [filename1]

    # # Final Data info
    # files = [file_1, file_2, file_3, file_4, file_5]
    # movnames = ["2.5", "5", "7.5", "10", "12.5"]
    # filenames = [filename1, filename2, filename3, filename4, filename5]
    # tilt = [2.5, 5.0, 7.5, 10.0, 12.5] * pi/180.0

    # files = [file_0, file_1, file_2, file_3, file_4, file_5]
    # movnames = ["neg5", "2.5", "5", "7.5", "10", "12.5"]
    # filenames = [filename0, filename1, filename2, filename3, filename4, filename5]
    # tilt = [-5.0, 2.5, 5.0, 7.5, 10.0, 12.5] * pi/180.0

    # Power gain info
    files = [file_1, file_2, file_3, file_4, file_5]
    movnames = ["2.5", "5", "7.5", "10", "12.5"]
    filenames = [filename1, filename2, filename3, filename4, filename5]
    tilt = [2.5, 5.0, 7.5, 10.0, 12.5] * pi/180.0
    slices = 7.0:1.0:12.0
    Uinf = 8.1      # m/s

    params = params_struct(Uinf, slices, tilt, filename1, filename2, filename3, filename4, filename5)

    # generate objective function wrapper
    obj_func!(g, x) = objective_vel_wrapper!(g, x, params)
    # obj_func!(g,x) = objective_vel_wrapper!(g, x, params)

    # initialize design variable Vector
    CTA = 27.2
    CTB = -0.78
    CTC = 10.5
    CTb = 0.4
    CTη = 0.008
    kz1 = -0.563
    kz2 = 0.108
    kz3 = 0.027
    ky1 = 0.048
    ky2 = 0.018
    sigy01 = 0.266
    sigz01 = 0.168
    sigz02 = -0.014
    sigz03 = 0.0419
    alpha_in = 0.02
    x0 = [CTA, CTB, CTC, CTη, CTb, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03, alpha_in]

    # IPOPT options
    # ip_options = Dict("max_iter" => 10000, "tol" => 1e-3)
    # solver = IPOPT(ip_options)
    snopt_options = Dict(
        "Derivative option" => 1,
        "Verify level" => 0,
        "Major optimality tolerance" => 1e-4,
        "Major iterations limit" => 5000,
        "Summary file" => "snopt-summary-tilt_testing.out",
        "Print file" => "snopt-print-tilt_testing.out")
    solver = SNOPT(options=snopt_options)

    options = Options(;solver, derivatives=ForwardAD())


    ng = 95
    lx = [-Inf, -Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf, -Inf,0.01, -(pi/2)+0.01]
    ux = [Inf, Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf, Inf,Inf, (pi/2)-0.01]
    # lg=[-Inf*ones(ng)]
    # ug=[zeros(ng)]
    # sigyzlow = 0.35
    # sigyzhigh = 0.7
    sigyzlow = 0.35
    sigyzhigh = 1.0
    lg=[-Inf*ones(30); -ones(5)*0.84; sigyzlow*ones(60)]
    ug=[0*ones(30); 0.5*ones(5); sigyzhigh*ones(60)]


    # optimize
    x0_initial = x0
    iterations = 10

    filename_opt = "optimal_solution_newc_ct_73.jld"
    fopt_optimal = 100000
    best = x0
    for i = 1:iterations
        print("iterations: ", i, "\n")
        if i == 1
            CTA = 27.2
            CTB = -0.78
            CTC = 10.5
            CTb = 0.4
            CTη = 0.008
            kz1 = -0.563
            kz2 = 0.108
            kz3 = 0.027
            ky1 = 0.048
            ky2 = 0.018
            sigy01 = 0.266
            sigz01 = 0.168
            sigz02 = -0.014
            sigz03 = 0.0419
            alpha_in = 0.00
            x0 = [CTA, CTB, CTC, CTη, CTb, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03, alpha_in]
        else
            xc = 1.0.-(rand(15)*.1)
            CTA = 27.2
            CTB = -0.78
            CTC = 10.5
            CTb = 0.4
            CTη = 0.008
            kz1 = -0.563
            kz2 = 0.108
            kz3 = 0.027
            ky1 = 0.048
            ky2 = 0.018
            sigy01 = 0.266
            sigz01 = 0.168
            sigz02 = -0.014
            sigz03 = 0.0419
            alpha_in = 0.00
            x0 = @. xc*[CTA, CTB, CTC, CTη, CTb, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03, alpha_in]
        end
        xopt, fopt, info, out = minimize(obj_func!, x0, ng, lx, ux, lg, ug, options)
        if fopt < fopt_optimal
            fopt_optimal = fopt
            best = xopt
            save(filename_opt, "xopt", best, "fopt", fopt_optimal)
            objective_vel_wrapper_plot(xopt, params)
        end
    end
end


function run_optimization_ipopt()
    file_0 = "/Users/jamescutler/Downloads/byu_tilt_runs_single/c_000_sp7_1turb_hNormal_D126_tilt_base/lite_data_000.ftr"
    file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_001_sp7_1turb_hNormal_D126_tilt_2.5/lite_data_001.ftr"
    file_2 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_004_sp7_1turb_hNormal_D126_tilt_5/lite_data_004.ftr"
    file_3 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_002_sp7_1turb_hNormal_D126_tilt_7.5/lite_data_002.ftr"
    file_4 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_005_sp7_1turb_hNormal_D126_tilt_10/lite_data_005.ftr"
    file_5 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_003_sp7_1turb_hNormal_D126_tilt_12.5/lite_data_003.ftr"

    # for power gains
    # filename0 = "analysis_smoothing_piece_wise_largeest_span_base_normal_n.jld"
    # filename1 = "analysis_smoothing_piece_wise_largeest_span_2.5_normal_n.jld"
    # filename2 = "analysis_smoothing_piece_wise_largeest_span_5_normal_n.jld"
    # filename3 = "analysis_smoothing_piece_wise_largeest_span_7.5_normal_n.jld"
    # filename4 = "analysis_smoothing_piece_wise_largeest_span_10_normal_n.jld"
    # filename5 = "analysis_smoothing_piece_wise_largeest_span_12.5_normal_n.jld"

    filename0 = "analysis_smoothing_piece_wise_largeest_span_base_normal_originalSOWFA.jld"
    filename1 = "analysis_smoothing_piece_wise_largeest_span_2.5_normal_originalSOWFA.jld"
    filename2 = "analysis_smoothing_piece_wise_largeest_span_5_normal_originalSOWFA.jld"
    filename3 = "analysis_smoothing_piece_wise_largeest_span_7.5_normal_originalSOWFA.jld"
    filename4 = "analysis_smoothing_piece_wise_largeest_span_10_normal_originalSOWFA.jld"
    filename5 = "analysis_smoothing_piece_wise_largeest_span_12.5_normal_originalSOWFA.jld"


    Gompertz_dist(x, η, b) = @. (b*η)*exp(η + (b*x) - (η*exp(10.0*b*x)))
    # y and z array
    y_array = 2245.0:0.5:2745.0
    z_array = 5.0:((304.99999999999994-5.0)/1000):304.99999999999994

    # Final data values
    # filename0 = "analysis_smoothing_piece_wise_largeest_span_base_normal_n_final.jld"
    # filename1 = "analysis_smoothing_piece_wise_largeest_span_2.5_normal_n_final.jld"
    # filename2 = "analysis_smoothing_piece_wise_largeest_span_5_normal_n_final.jld"
    # filename3 = "analysis_smoothing_piece_wise_largeest_span_7.5_normal_n_final.jld"
    # filename4 = "analysis_smoothing_piece_wise_largeest_span_10_normal_n_final.jld"
    # filename5 = "analysis_smoothing_piece_wise_largeest_span_12.5_normal_n_final.jld"


    # filename1 = "analysis_25.jld"
    # filename2 = "analysis_neg15.jld"
    # filename1 = "analysis_smoothing_piece_wise_largeest_span_25_normal.jld"


    # files = [file_1]
    # movnames = ["25"]
    # filenames = [filename1]

    # # Final Data info
    # files = [file_1, file_2, file_3, file_4, file_5]
    # movnames = ["2.5", "5", "7.5", "10", "12.5"]
    # filenames = [filename1, filename2, filename3, filename4, filename5]
    # tilt = [2.5, 5.0, 7.5, 10.0, 12.5] * pi/180.0

    # files = [file_0, file_1, file_2, file_3, file_4, file_5]
    # movnames = ["neg5", "2.5", "5", "7.5", "10", "12.5"]
    # filenames = [filename0, filename1, filename2, filename3, filename4, filename5]
    # tilt = [-5.0, 2.5, 5.0, 7.5, 10.0, 12.5] * pi/180.0

    # Power gain info
    files = [file_1, file_2, file_3, file_4, file_5]
    movnames = ["2.5", "5", "7.5", "10", "12.5"]
    filenames = [filename1, filename2, filename3, filename4, filename5]
    tilt = [2.5, 5.0, 7.5, 10.0, 12.5] * pi/180.0
    slices = 7.0:1.0:12.0
    Uinf = 8.1      # m/s

    params = params_struct(Uinf, slices, tilt, filename1, filename2, filename3, filename4, filename5)

    # generate objective function wrapper
    obj_func!(g,x) = objective_vel_wrapper!(g, x, params)

    # initialize design variable Vector
    CTA = 27.2
    CTB = -0.78
    CTC = 10.5
    CTb = 0.4
    CTη = 0.008
    kz1 = -0.563
    kz2 = 0.108
    kz3 = 0.027
    ky1 = 0.048
    ky2 = 0.018
    sigy01 = 0.266
    sigz01 = 0.168
    sigz02 = -0.014
    sigz03 = 0.0419
    x0 = [CTA, CTB, CTC, CTη, CTb, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03]

    # IPOPT options
    ip_options = Dict("max_iter" => 10000, "tol" => 1e-2)
    solver = IPOPT(ip_options)

    options = Options(solver=solver, derivatives=ForwardAD())


    ng = 95
    lx = [-Inf, -Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf, -Inf,0.01]
    ux = [Inf, Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf, Inf,Inf]
    # lg=[-Inf*ones(ng)]
    # ug=[zeros(ng)]
    lg=[-Inf*ones(30); -ones(5)*0.85; 0.35*ones(60)]
    ug=[0*ones(30); 0.5*ones(5); 0.7*ones(60)]


    # optimize
    x0_initial = x0
    iterations = 100

    filename_opt = "optimal_solution_newc.jld"
    fopt_optimal = 100000
    best = x0
    for i = 1:iterations
        CTA = 27.2
        CTB = -0.78
        CTC = 10.5
        CTb = 0.4
        CTη = 0.008
        kz1 = -0.563
        kz2 = 0.108
        kz3 = 0.027
        ky1 = 0.048
        ky2 = 0.018
        sigy01 = 0.266
        sigz01 = 0.168
        sigz02 = -0.014
        sigz03 = 0.0419
        x0 = [CTA, CTB, CTC, CTη, CTb, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03]
        xopt, fopt, info, out = minimize(obj_func!, x0, ng, lx, ux, lg, ug, options)
        if fopt < fopt_optimal
            fopt_optimal = fopt
            best = xopt
            save(filename_opt, "xopt", best, "fopt", fopt_optimal)
            objective_vel_wrapper_plot(xopt, params)
        end
    end
end

function check_with_original(x0_opt)
    # compare to original
    file_0 = "/Users/jamescutler/Downloads/byu_tilt_runs_single/c_000_sp7_1turb_hNormal_D126_tilt_base/lite_data_000.ftr"
    file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_001_sp7_1turb_hNormal_D126_tilt_2.5/lite_data_001.ftr"
    file_2 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_004_sp7_1turb_hNormal_D126_tilt_5/lite_data_004.ftr"
    file_3 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_002_sp7_1turb_hNormal_D126_tilt_7.5/lite_data_002.ftr"
    file_4 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_005_sp7_1turb_hNormal_D126_tilt_10/lite_data_005.ftr"
    file_5 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_003_sp7_1turb_hNormal_D126_tilt_12.5/lite_data_003.ftr"

    filename0 = "analysis_smoothing_piece_wise_largeest_span_base_normal_originalSOWFA.jld"
    filename1 = "analysis_smoothing_piece_wise_largeest_span_2.5_normal_originalSOWFA.jld"
    filename2 = "analysis_smoothing_piece_wise_largeest_span_5_normal_originalSOWFA.jld"
    filename3 = "analysis_smoothing_piece_wise_largeest_span_7.5_normal_originalSOWFA.jld"
    filename4 = "analysis_smoothing_piece_wise_largeest_span_10_normal_originalSOWFA.jld"
    filename5 = "analysis_smoothing_piece_wise_largeest_span_12.5_normal_originalSOWFA.jld"


    Gompertz_dist(x, η, b) = @. (b*η)*exp(η + (b*x) - (η*exp(10.0*b*x)))
    # y and z array
    y_array = 2245.0:0.5:2745.0
    z_array = 5.0:((304.99999999999994-5.0)/1000):304.99999999999994

    # Power gain info
    files = [file_1, file_2, file_3, file_4, file_5]
    movnames = ["2.5", "5", "7.5", "10", "12.5"]
    filenames = [filename1, filename2, filename3, filename4, filename5]
    tilt = [2.5, 5.0, 7.5, 10.0, 12.5] * pi/180.0
    slices = 7.0:1.0:12.0
    Uinf = 8.1      # m/s

    params = params_struct(Uinf, slices, tilt, filename1, filename2, filename3, filename4, filename5)
    # compare to original
    CTA = 27.2
    CTB = -0.78
    CTC = 10.5
    CTb = 0.4
    CTη = 0.008
    kz1 = -0.563
    kz2 = 0.108
    kz3 = 0.027
    ky1 = 0.048
    ky2 = 0.018
    sigy01 = 0.266
    sigz01 = 0.168
    sigz02 = -0.014
    sigz03 = 0.0419
    alpha_in = 0.00
    x0 = [CTA, CTB, CTC, CTη, CTb, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03, alpha_in]
    print("x0: ", x0, "\n")
    print("params: ", params, "\n")
    # objective_vel_wrapper_plot(x0, params)

    objective_vel_wrapper_plot_difference(x0, x0_opt, params)

end

function check_original()
    # compare to original
    file_0 = "/Users/jamescutler/Downloads/byu_tilt_runs_single/c_000_sp7_1turb_hNormal_D126_tilt_base/lite_data_000.ftr"
    file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_001_sp7_1turb_hNormal_D126_tilt_2.5/lite_data_001.ftr"
    file_2 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_004_sp7_1turb_hNormal_D126_tilt_5/lite_data_004.ftr"
    file_3 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_002_sp7_1turb_hNormal_D126_tilt_7.5/lite_data_002.ftr"
    file_4 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_005_sp7_1turb_hNormal_D126_tilt_10/lite_data_005.ftr"
    file_5 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_003_sp7_1turb_hNormal_D126_tilt_12.5/lite_data_003.ftr"

    filename0 = "analysis_smoothing_piece_wise_largeest_span_base_normal_originalSOWFA.jld"
    filename1 = "analysis_smoothing_piece_wise_largeest_span_2.5_normal_originalSOWFA.jld"
    filename2 = "analysis_smoothing_piece_wise_largeest_span_5_normal_originalSOWFA.jld"
    filename3 = "analysis_smoothing_piece_wise_largeest_span_7.5_normal_originalSOWFA.jld"
    filename4 = "analysis_smoothing_piece_wise_largeest_span_10_normal_originalSOWFA.jld"
    filename5 = "analysis_smoothing_piece_wise_largeest_span_12.5_normal_originalSOWFA.jld"


    Gompertz_dist(x, η, b) = @. (b*η)*exp(η + (b*x) - (η*exp(10.0*b*x)))
    # y and z array
    y_array = 2245.0:0.5:2745.0
    z_array = 5.0:((304.99999999999994-5.0)/1000):304.99999999999994

    # Power gain info
    files = [file_1, file_2, file_3, file_4, file_5]
    movnames = ["2.5", "5", "7.5", "10", "12.5"]
    filenames = [filename1, filename2, filename3, filename4, filename5]
    tilt = [2.5, 5.0, 7.5, 10.0, 12.5] * pi/180.0
    slices = 7.0:1.0:12.0
    Uinf = 8.1      # m/s

    params = params_struct(Uinf, slices, tilt, filename1, filename2, filename3, filename4, filename5)
    # compare to original
    CTA = 27.2
    CTB = -0.78
    CTC = 10.5
    CTb = 0.4
    CTη = 0.008
    kz1 = -0.563
    kz2 = 0.108
    kz3 = 0.027
    ky1 = 0.048
    ky2 = 0.018
    sigy01 = 0.266
    sigz01 = 0.168
    sigz02 = -0.014
    sigz03 = 0.0419
    alpha_in = 0.00
    x0 = [CTA, CTB, CTC, CTη, CTb, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03, alpha_in]
    print("x0: ", x0, "\n")
    print("params: ", params, "\n")
    # objective_vel_wrapper_plot(x0, params)

    objective_vel_wrapper_plot(x0, params)

end

Gompertz_dist(x, η, b) = @. (b*η)*exp(η + (b*x) - (η*exp(10.0*b*x)))
# run_optimization_SNOPT()

filename_opt = "optimal_solution_newc_ct_73.jld"
file_detail =  "objective_best_opt.jld"     # details of no optimization
file_detail_original = "objective_best_opt_x0.jld"          # details of with optimization
x = load(filename_opt, "xopt")
fopt = load(filename_opt, "fopt")
slices = 7.0:1.0:12.0
tilt = [2.5, 5.0, 7.5, 10.0, 12.5]*pi/180
fopt_detailed = load(file_detail, "f_detail")
fopt_detailed_original = load(file_detail_original, "f_detail")
f_diff = @. (fopt_detailed - fopt_detailed_original)
fontylabel = text("", 12).font
fontylabel.rotation=0
font25_c = text("", 12, color=:lightskyblue2).font
font5_c = text("", 12,color=:skyblue2).font
font75_c = text("", 12,color=:steelblue2).font
font10_c = text("", 12,color=:dodgerblue).font
font125_c = text("", 12,color=:dodgerblue3).font

font25_o = text("", 12, color=:salmon1).font
font5_o= text("", 12,color=:tomato).font
font75_o = text("", 12,color=:firebrick4).font
font10_o = text("", 12,color=:darkred).font
font125_o = text("", 12,color=:red3).font
# fontylabel = text("", 11,color=:black).font
# fopt_detailed_divide = fopt_detailed/(31*51)
X_ax = 12.3
plot(slices, fopt_detailed_original[1], color=:lightskyblue2, label="2.5")
plot!(slices, fopt_detailed[1], seriestype=:scatter, color=:salmon1)
plot!(slices, fopt_detailed_original[2], color=:skyblue2, label="5.0")
plot!(slices, fopt_detailed[2], seriestype=:scatter, color=:tomato, label="5.0")
plot!(slices, fopt_detailed_original[3], color=:steelblue2, label="7.5")
plot!(slices, fopt_detailed[3], seriestype=:scatter, color=:firebrick4, label="7.5")
plot!(slices, fopt_detailed_original[4], color=:dodgerblue, label="10.0", xlabel="x/D (Downstream Distance)")
plot!(slices, fopt_detailed[4], seriestype=:scatter, color=:darkred)
plot!(slices, fopt_detailed_original[5], color=:dodgerblue3, label="12.5",legend=:false,xlimit = [7, 12.5], legendfontsize=10, grid=false, xticks=[7, 8, 9, 10, 11, 12], dpi=300)
plot!(slices, fopt_detailed[5], seriestype=:scatter, color=:red3)
plot!(annotations = ([X_ax], fopt_detailed_original[1][end]+0.0018,text("2.5°", font25_c)))
plot!(annotations = ([X_ax], fopt_detailed_original[2][end]+0.00017, text("5.0°", font5_c)))
plot!(annotations = ([X_ax], fopt_detailed_original[3][end]-0.0005, text("7.5°", font75_c)))
plot!(annotations = ([X_ax], fopt_detailed_original[4][end], text("10.0°", font10_c)))
plot!(annotations = ([X_ax], fopt_detailed_original[5][end], text("12.5°", font125_c)), left_margin=15Plots.mm, transparent=true, tickfontsize=12, guidefontsize=12)
plot!(annotations = ([X_ax], fopt_detailed[1][end]+0.0018,text("2.5°", font25_o)))
plot!(annotations = ([X_ax], fopt_detailed[2][end]+0.00017, text("5.0°", font5_o)))
plot!(annotations = ([X_ax], fopt_detailed[3][end]-0.0005, text("7.5°", font75_o)))
plot!(annotations = ([X_ax], fopt_detailed[4][end], text("10.0°", font10_o)))
plot!(annotations = ([X_ax], fopt_detailed[5][end], text("12.5°", font125_o)), left_margin=15Plots.mm, transparent=true, tickfontsize=12, guidefontsize=12)
# plot!(annotations = ([5.9], 0.142,text(L"\sqrt{\frac{\sum_{i=1}^{N_y} \sum_{j=1}^{N_z} |V_{ij} - Vm_{ij}|^2}{N_y N_z}}", fontylabel)))
plot!(annotations = ([5.9], 0.142,text("RMS", fontylabel)))
savefig("optimized_objective_pow.pdf")
# plot!(annotations = ([6.3], 5.7,text(L"\sum_{i=1}^{Y} \sum_{j=1}^{Z} \sqrt{abs{V_{ij} - V_m_{ij}}^2}", fontylabel)))

# # run original x0
# # check_original()

# plot(slices, -f_diff[1], label="2.5")
# plot!(slices, -f_diff[2], label="5.0")
# plot!(slices, -f_diff[3], label="7.5")
# plot!(slices, -f_diff[4], label="10.0")
# plot!(slices, -f_diff[5], label="12.5")

font25_c = text("", 12, color=:lightskyblue2).font
font5_c = text("", 12,color=:skyblue2).font
font75_c = text("", 12,color=:steelblue2).font
font10_c = text("", 12,color=:dodgerblue).font
font125_c = text("", 12,color=:dodgerblue3).font
RMS = "RMS"

RMS1 = (f_diff[1]./fopt_detailed_original[1])*100
RMS2 = (f_diff[2]./fopt_detailed_original[2])*100
RMS3 = (f_diff[3]./fopt_detailed_original[3])*100
RMS4 = (f_diff[4]./fopt_detailed_original[4])*100
RMS5 = (f_diff[5]./fopt_detailed_original[5])*100

"""Plot diff"""
plot(slices, RMS1, color=:lightskyblue2, label="2.5")
plot!(slices, RMS2, color=:skyblue2, label="5.0")
plot!(slices, RMS3, color=:steelblue2, label="7.5")
plot!(slices, RMS4, color=:dodgerblue, label="10.0", xlabel="x/D (Downstream Distance)")
plot!(slices, RMS5, color=:dodgerblue3, label="12.5",legend=:false,xlimit = [7, 12.5], legendfontsize=10, grid=false, xticks=[7, 8, 9, 10, 11, 12], dpi=300)
plot!(annotations = ([X_ax], RMS1[end]+0.2,text("2.5°", font25_c)))
plot!(annotations = ([X_ax], RMS2[end], text("5.0°", font5_c)))
plot!(annotations = ([X_ax], RMS3[end]-0.2, text("7.5°", font75_c)))
plot!(annotations = ([X_ax], RMS4[end], text("10.0°", font10_c)))
plot!(annotations = ([X_ax], RMS5[end], text("12.5°", font125_c)), left_margin=20Plots.mm, transparent=true, tickfontsize=12, guidefontsize=12)
plot!(annotations = ([6.0], -7.5,text("\\Delta  RMS %", fontylabel)))
savefig("diff_objective_pow.pdf")
# # plot!(annotations = ([6.3], 0.5,text(L"|Objective_{0} - Objective_{m}|", fontylabel)))
# # plot!(annotations = ([6.3], 5.7,text(L"\sum_{i=1}^{Y} \sum_{j=1}^{Z} \sqrt{abs{V_{ij} - V_m_{ij}}^2}", fontylabel)))


# # plot(slices, fopt_detailed_original[1], label="2.5_o")
# # plot!(slices, fopt_detailed_original[2], label="5.0_o")
# # plot!(slices, fopt_detailed_original[3], label="7.5_o")
# # plot!(slices, fopt_detailed_original[4], label="10.0_o")
# # plot!(slices, fopt_detailed_original[5], label="12.5_o")

plot(slices, fopt_detailed[1], color=:lightskyblue2, label="2.5")
plot!(slices, fopt_detailed[2], color=:skyblue2, label="5.0")
plot!(slices, fopt_detailed[3], color=:steelblue2, label="7.5")
plot!(slices, fopt_detailed[4], color=:dodgerblue, label="10.0", xlabel="x/D (Downstream Distance)")
plot!(slices, fopt_detailed[5], color=:dodgerblue3, label="12.5",legend=:false,xlimit = [7, 12.5], legendfontsize=10, grid=false, xticks=[7, 8, 9, 10, 11, 12], dpi=300)
plot!(annotations = ([X_ax], fopt_detailed[1][end]-0.001,text("2.5°", font25_c)))
plot!(annotations = ([X_ax], fopt_detailed[2][end]-0.002, text("5.0°", font5_c)))
plot!(annotations = ([X_ax], fopt_detailed[3][end]+0.002, text("7.5°", font75_c)))
plot!(annotations = ([X_ax], fopt_detailed[4][end]-0.002, text("10.0°", font10_c)))
plot!(annotations = ([X_ax], fopt_detailed[5][end], text("12.5°", font125_c)), left_margin=15Plots.mm, transparent=true, tickfontsize=12, guidefontsize=12)
plot!(annotations = ([5.9], 0.142,text("RMS", fontylabel)))
savefig("original_objective_pow.pdf")

# plot(slices, sigy_2_5,color=:lightskyblue2, label="2.5°")
# plot!(slice_fine[6:end], sigmaY_2_5[11:end-M], seriestype=:scatter, color=:lightskyblue2, label="2.5°")
# plot!(slices, sigy_5,color=:skyblue2, label="5°")
# plot!(slice_fine[6:end], sigmaY_5[11:end-M], seriestype=:scatter, color=:skyblue2, label="2.5°")
# plot!(slices, sigy_7_5,color=:steelblue2, label="7.5°")
# plot!(slice_fine[6:end], sigmaY_7_5[11:end-M], seriestype=:scatter, color=:steelblue2, label="2.5°")
# plot!(slices, sigy_10,color=:dodgerblue,label="10°")
# plot!(slice_fine[6:end], sigmaY_10[11:end-M], seriestype=:scatter, color=:dodgerblue, label="2.5°")
# plot!(slices, sigy_12_5,color=:dodgerblue3, label="12.5°", xlabel="x/D (Downstream Distance)", xrotation = 0,legend=:false,xlimit = [7, 12.5], legendfontsize=10, grid=false, xticks=[7, 8, 9, 10, 11, 12], dpi=300)
# plot!(slice_fine[6:end], sigmaY_12_5[11:end-M], seriestype=:scatter, color=:dodgerblue3, label="2.5°")
# plot!(annotations = ([X_ax1], sigy_2_5_fine[3]+Z_fix-(sigy_2_5_fine[3]-sigy_2_5_fine[1])/2, text("$ky2_5", font25)))
# plot!(Xline, Y_2_5, color=:lightskyblue2)
# plot!(annotations = ([X_ax1], sigy_5_fine[3]+Z_fix-0.001-(sigy_5_fine[3]-sigy_5_fine[1])/2, text("$ky_5", font5)))
# plot!(Xline, Y_5, color=:skyblue2)
# plot!(annotations = ([X_ax1], sigy_7_5_fine[3]+Z_fix+0.001-(sigy_7_5_fine[3]-sigy_7_5_fine[1])/2, text("$ky7_5", font75)))
# plot!(Xline, Y_7_5, color=:steelblue2)
# plot!(annotations = ([X_ax1], sigy_10_fine[3]+Z_fix+0.002-(sigy_10_fine[3]-sigy_10_fine[1])/2, text("$ky_10", font10)))
# plot!(Xline, Y_10, color=:dodgerblue)
# plot!(annotations = ([X_ax1], sigy_12_5_fine[3]+Z_fix+0.002-(sigy_12_5_fine[3]-sigy_12_5_fine[1])/2, text("$ky12_5", font125)))
# plot!(Xline, Y_12_5, color=:dodgerblue3)
# plot!(annotations = ([X_ax2], [Y_ax1], "1", 9))
# plot!(annotations = ([Y_ax], 0.505,text("{\\sigma_y/D}", fontylabel)))
# plot!(annotations = ([X_ax], sigy_2_5[end],text("2.5°", font25_c)))
# plot!(annotations = ([X_ax], sigy_5[end], text("5.0°", font5_c)))
# plot!(annotations = ([X_ax], sigy_7_5[end], text("7.5°", font75_c)))
# plot!(annotations = ([X_ax], sigy_10[end], text("10.0°", font10_c)))
# plot!(annotations = ([X_ax], sigy_12_5[end], text("12.5°", font125_c)), left_margin=12Plots.mm, transparent=true, tickfontsize=12, guidefontsize=12)
# savefig("sigma_y_fit_opt.pdf")

























# # load measured points:
# # filename0 = "analysis_smoothing_piece_wise_largeest_span_base_normal_n_final.jld"
# filename1 = "analysis_smoothing_piece_wise_largeest_span_2.5_normal_n_final.jld"
# filename2 = "analysis_smoothing_piece_wise_largeest_span_5_normal_n_final.jld"
# filename3 = "analysis_smoothing_piece_wise_largeest_span_7.5_normal_n_final.jld"
# filename4 = "analysis_smoothing_piece_wise_largeest_span_10_normal_n_final.jld"
# filename5 = "analysis_smoothing_piece_wise_largeest_span_12.5_normal_n_final.jld"

# # sigmaY_neg_5 = load(filename0, "sigy")
# # sigmaZ_neg_5 = load(filename0, "sigz")
# # defZ_neg_5 = load(filename0, "defz")
# # defY_neg_5 = load(filename0, "defy")
# # sigmaZ_g_neg_5 = load(filename0, "sigz_g")
# # cup_neg_5 = load(filename0, "cup")
# # cdown_neg_5 = load(filename0, "cdown")
# # power_neg_5 = load(filename0, "power")
# # power_neg_5_up = load(filename0, "power_up")
# # avgV_neg_5 = load(filename0, "avgV")
# # defZ_neg_5_analytical = 90.0.+deflection_est((-5*pi/180), X_D)*126.0
# # # fix power
# # power_neg_5_up = @. (power_neg_5_up/Cp_delete)*CP_interp(avgV_neg_5)

# sigmaY_2_5 = load(filename1, "sigy")
# sigmaZ_2_5 = load(filename1, "sigz")
# defZ_2_5 = load(filename1, "defz")
# defY_2_5 = load(filename1, "defy")
# sigmaZ_g_2_5 = load(filename1, "sigz_g")
# cup_2_5 = load(filename1, "cup")
# cdown_2_5 = load(filename1, "cdown")
# power_2_5 = load(filename1, "power")
# power_2_5_up = load(filename1, "power_up")
# avgV_2_5 = load(filename1, "avgV")
# defZ_2_5_analytical = 90.0 .+ deflection_est((2.5*pi/180), X_D)*126.0
# # fix power
# power_2_5_up = @. (power_2_5_up/Cp_delete)*CP_interp(avgV_2_5)

# sigmaY_5 = load(filename2, "sigy")
# sigmaZ_5 = load(filename2, "sigz")
# defZ_5 = load(filename2, "defz")
# defY_5 = load(filename2, "defy")
# sigmaZ_g_5 = load(filename2, "sigz_g")
# cup_5 = load(filename2, "cup")
# cdown_5 = load(filename2, "cdown")
# power_5 = load(filename2, "power")
# power_5_up = load(filename2, "power_up")
# avgV_5 = load(filename2, "avgV")
# defZ_5_analytical = 90.0 .+ deflection_est((5*pi/180), X_D)*126.0
# # fix power
# power_5_up = @. (power_5_up/Cp_delete)*CP_interp(avgV_5)

# sigmaY_7_5 = load(filename3, "sigy")
# sigmaZ_7_5 = load(filename3, "sigz")
# defZ_7_5 = load(filename3, "defz")
# defY_7_5 = load(filename3, "defy")
# sigmaZ_g_7_5 = load(filename3, "sigz_g")
# cup_7_5 = load(filename3, "cup")
# cdown_7_5 = load(filename3, "cdown")
# power_7_5 = load(filename3, "power")
# power_7_5_up = load(filename3, "power_up")
# avgV_7_5 = load(filename3, "avgV")
# defZ_7_5_analytical = 90.0 .+ deflection_est((7.5*pi/180), X_D)*126.0
# # fix power
# power_7_5_up = @. (power_7_5_up/Cp_delete)*CP_interp(avgV_7_5)

# sigmaY_10 = load(filename4, "sigy")
# sigmaZ_10 = load(filename4, "sigz")
# defZ_10 = load(filename4, "defz")
# defY_10 = load(filename4, "defy")
# sigmaZ_g_10 = load(filename4, "sigz_g")
# cup_10 = load(filename4, "cup")
# cdown_10 = load(filename4, "cdown")
# power_10 = load(filename4, "power")
# power_10_up = load(filename4, "power_up")
# avgV_10 = load(filename4, "avgV")
# defZ_10_analytical = 90.0 .+ deflection_est((10*pi/180), X_D)*126.0
# # fix power
# power_10_up = @. (power_10_up/Cp_delete)*CP_interp(avgV_10)

# sigmaY_12_5 = load(filename5, "sigy")
# sigmaZ_12_5 = load(filename5, "sigz")
# defZ_12_5 = load(filename5, "defz")
# defY_12_5 = load(filename5, "defy")
# sigmaZ_g_12_5 = load(filename5, "sigz_g")
# cup_12_5 = load(filename5, "cup")
# cdown_12_5 = load(filename5, "cdown")
# power_12_5 = load(filename5, "power")
# power_12_5_up = load(filename5, "power_up")
# avgV_12_5 = load(filename5, "avgV")
# defZ_12_5_analytical = 90.0 .+ deflection_est((12.5*pi/180), X_D)*126.0
# # fix power
# power_12_5_up = @. (power_12_5_up/Cp_delete)*CP_interp(avgV_12_5)

# default(titlefont= ("times"), guidefont=("times"), tickfont=("times"))


# # plot(slices, fopt_detailed_divide[1], label="2.5")
# # plot!(slices, fopt_detailed_divide[2], label="5.0")
# # plot!(slices, fopt_detailed_divide[3], label="7.5")
# # plot!(slices, fopt_detailed_divide[4], label="10.0")
# # plot!(slices, fopt_detailed_divide[5], label="12.5")


# CTA = x[1]
# CTB = x[2]
# CTC = x[3]
# CTη = x[4]
# CTb = x[5]
# # for kz and ky
# kz1 = x[6]
# kz2 = x[7]
# kz3 = x[8]
# ky1 = x[9]
# ky2 = x[10]
# # for sigy0 and sigz0
# sigy01 = x[11]
# sigz01 = x[12]
# sigz02 = x[13]
# sigz03 = x[14]
# # alpha_opt
# alpha_in = x[15]

# A = 27.2
# B = -0.78
# C = 10.5
# C_T_new_adjusted = 7.620929940139257175e-01 .+ 1A*Gompertz_dist((tilt*C).+B, 0.008, 0.4)

# tilt = [2.5, 5.0, 7.5, 10.0, 12.5]*pi/180.0
# # check CT
# C_T_opt = CTA*Gompertz_dist((tilt*CTC).+CTB, CTη, CTb)
# plot(tilt, C_T_new_adjusted)
# plot!(tilt, C_T_opt)

# # check kz
# D = 126
# slices = 6:0.2:15
# fontsz = 13
# linew = 2
# marks = 5
# fontkz_c = text("", 12,color=:black).font
# fontky_c = text("", 12,color=:dodgerblue).font
# kz_pred = @. -0.563*(tilt^2) + 0.108*tilt + 0.027
# kz_opt = @. kz1*(tilt^2) + kz2*tilt + kz3
# kz1_p = round(kz1, digits=3)
# kz2_p = round(kz2, digits=3)
# kz3_p = round(kz3, digits=3)
# plot(tilt*(180/pi), kz_pred, color=:dodgerblue)
# plot!(tilt*(180/pi), kz_opt, color=:black, xlabel="Turbine Tilt (degrees)", grid=false, xticks=[2.5, 5.0, 7.5, 10.0, 12.5])

# # check ky
# ky_pred = @. 0.048*tilt + 0.018
# ky_opt = @. ky1*tilt + ky2
# ky1_p = round(ky1, digits=3)
# ky2_p = round(ky2, digits=3)

# kz1_c = -0.563
# kz2_c = 0.108
# kz3_c = 0.027
# ky1_c = 0.048
# ky2_c = 0.018
# plot!(tilt*(180/pi), ky_pred, color=:dodgerblue, xtickfontsize=10, ytickfontsize=10, xguidefontsize=12, yguidefontsize=12)
# plot!(tilt*(180/pi), ky_opt, color=:black, grid=:false, legend=:false, ylimits=[0.0200, 0.0325], left_margin=18Plots.mm, yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz,foreground_color_legend = nothing)
# plot!(annotations = ([6.25], 0.0295, text("kz = $kz1_p\\gamma^{2} + $kz2_p\\gamma + $kz3_p", fontkz_c)))
# plot!(annotations = ([4.4], 0.0248,text("ky = $ky1_p\\gamma + $ky2_p", fontkz_c)))
# plot!(annotations = ([9.1], 0.0325, text("kz = $kz1_c\\gamma^{2} + $kz2_c\\gamma + $kz3_c", fontky_c)))
# plot!(annotations = ([7.2], 0.022,text("ky = $ky1_c\\gamma + $ky2_c", fontky_c)))
# plot!(annotations = ([-.1], 0.0259,text("k-value", fontylabel)))
# savefig("kykz_both_opt_pow.pdf")
# # check sigy0


# # check sigz0
# sigz_pred = @. 0.168 - 0.014*log(tilt - 0.0419)
# sigz0_opt = @. sigz01 + sigz02*log(tilt + sigz03)
# sigz01_p = round(sigz01, digits=3)
# sigz02_p = round(sigz02, digits=3)
# sigz03_p = round(sigz03, digits=3)
# plot(tilt*(180/pi), sigz_pred, label="pred", color=:dodgerblue, xlabel="Tilt (degrees)", left_margin=18Plots.mm, yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz,foreground_color_legend = nothing)
# plot!(tilt*(180/pi), sigz0_opt, label="opt", color=:black, legend=:false, grid=:false, xticks=[2.5, 5.0, 7.5, 10.0, 12.5])
# plot!(annotations = ([5.6], 0.196, text("\\sigma_{z0} = 0.168 - 0.014 ln (\\gamma - 0.0419)", fontky_c)))
# plot!(annotations = ([8.5], 0.22, text("\\sigma_{z0} = $sigz01_p - $sigz02_p ln (\\gamma + $sigz03_p)", fontkz_c)))
# plot!(annotations = ([0.5], 0.225,text("\\sigma_0", fontylabel)))
# savefig("sigz0_opt_pow.pdf")
# """for Plotting sigy over varying tilt angles"""

# slices = 7:1.0:12.0

# slice_fine = 7:0.2:12

# sigy_2_5, sigz_2_5 = sigysigz_solve(slices, tilt[1], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
# sigy_5, sigz_5 = sigysigz_solve(slices, tilt[2], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
# sigy_7_5, sigz_7_5 = sigysigz_solve(slices, tilt[3], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
# sigy_10, sigz_10 = sigysigz_solve(slices, tilt[4], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
# sigy_12_5, sigz_12_5 = sigysigz_solve(slices, tilt[5], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)

# sigy_2_5_fine, sigz_2_5_fine = sigysigz_solve(slice_fine, tilt[1], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
# sigy_5_fine, sigz_5_fine = sigysigz_solve(slice_fine, tilt[2], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
# sigy_7_5_fine, sigz_7_5_fine = sigysigz_solve(slice_fine, tilt[3], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
# sigy_10_fine, sigz_10_fine = sigysigz_solve(slice_fine, tilt[4], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
# sigy_12_5_fine, sigz_12_5_fine = sigysigz_solve(slice_fine, tilt[5], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)


# ky2_5, kz2_5 = kykzsolve(slices, tilt[1], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
# ky_5, kz_5 = kykzsolve(slices, tilt[2], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
# ky7_5, kz7_5 = kykzsolve(slices, tilt[3], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
# ky_10, kz_10 = kykzsolve(slices, tilt[4], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
# ky12_5, kz12_5 = kykzsolve(slices, tilt[5], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)

# ky2_5 = round(ky2_5, digits = 3)
# ky_5 = round(ky_5, digits = 3)
# ky7_5 = round(ky7_5, digits = 3)
# ky_10 = round(ky_10, digits = 3)
# ky12_5 = round(ky12_5, digits = 3)

# kz2_5 = round(kz2_5, digits = 3)
# kz_5 = round(kz_5, digits = 3)
# kz7_5 = round(kz7_5, digits = 3)
# kz_10 = round(kz_10, digits = 3)
# kz12_5 = round(kz12_5, digits = 3)

# X_ax = 12.4
# X_ax1 = 7.65
# X_ax2 = 7.2
# Y_ax1 = 0.338
# Xline = [7, 7.4, 7.4]
# Y_2_5 = [sigy_2_5_fine[1], sigy_2_5_fine[1], sigy_2_5_fine[3]]
# Y_5 = [sigy_5_fine[1], sigy_5_fine[1], sigy_5_fine[3]]
# Y_7_5 = [sigy_7_5_fine[1], sigy_7_5_fine[1], sigy_7_5_fine[3]]
# Y_10 = [sigy_10_fine[1], sigy_10_fine[1], sigy_10_fine[3]]
# Y_12_5 = [sigy_12_5_fine[1], sigy_12_5_fine[1], sigy_12_5_fine[3]]

# Z_2_5 = [sigz_2_5_fine[1], sigz_2_5_fine[1], sigz_2_5_fine[3]]
# Z_5 = [sigz_5_fine[1], sigz_5_fine[1], sigz_5_fine[3]]
# Z_7_5 = [sigz_7_5_fine[1], sigz_7_5_fine[1], sigz_7_5_fine[3]]
# Z_10 = [sigz_10_fine[1], sigz_10_fine[1], sigz_10_fine[3]]
# Z_12_5 = [sigz_12_5_fine[1], sigz_12_5_fine[1], sigz_12_5_fine[3]]

# # Coloring the tilt values
# font25_c = text("", 12, color=:lightskyblue2).font
# font5_c = text("", 12,color=:skyblue2).font
# font75_c = text("", 12,color=:steelblue2).font
# font10_c = text("", 12,color=:dodgerblue).font
# font125_c = text("", 12,color=:dodgerblue3).font
# fontylabel = text("", 12,color=:black).font

# # Rotating annotations
# font25 = text("", 9).font
# font25.rotation=23.5
# font5 = text("", 9).font
# font5.rotation=23
# font75 = text("", 9).font
# font75.rotation=22
# font10 = text("", 9).font
# font10.rotation=21
# font125 = text("", 9).font
# font125.rotation=20
# Y_ax = 6.0
# Z_fix = 0.005
# M = 15
# # # # Plot
# plot(slices, sigz_2_5,color=:lightskyblue2, label="2.5°")
# plot!(slice_fine[6:end], sigmaZ_2_5[11:end-M], seriestype=:scatter, color=:lightskyblue2, label="2.5°")
# plot!(slices, sigz_5,color=:skyblue2, label="5.0°")
# plot!(slice_fine[6:end], sigmaZ_5[11:end-M], seriestype=:scatter, color=:skyblue2, label="2.5°")
# plot!(slices, sigz_7_5,color=:steelblue2, label="7.5°")
# plot!(slice_fine[6:end], sigmaZ_7_5[11:end-M], seriestype=:scatter, color=:steelblue2, label="2.5°")
# plot!(slices, sigz_10,color=:dodgerblue,label="10.0°")
# plot!(slice_fine[6:end], sigmaZ_10[11:end-M], seriestype=:scatter, color=:dodgerblue, label="2.5°")
# plot!(slices, sigz_12_5,color=:dodgerblue3, label="12.5°", xlabel="x/D (Downstream Distance)", xrotation = 0,legend=:false,xlimit = [7, 12.5], legendfontsize=10, grid=false, xticks=[7, 8, 9, 10, 11, 12], dpi=300)
# plot!(slice_fine[6:end], sigmaZ_12_5[11:end-M], seriestype=:scatter, color=:dodgerblue3, label="2.5°")
# plot!(annotations = ([X_ax1-0.05], sigz_2_5_fine[3]+Z_fix+0.012-(sigz_2_5_fine[3]-sigy_2_5_fine[1])/2, text("$kz2_5", font25)))
# plot!(Xline, Z_2_5, color=:lightskyblue2)
# plot!(annotations = ([X_ax1], sigz_5_fine[3]+Z_fix+0.001-(sigz_5_fine[3]-sigz_5_fine[1])/2, text("$kz_5", font5)))
# plot!(Xline, Z_5, color=:skyblue2)
# plot!(annotations = ([X_ax1], sigz_7_5_fine[3]+Z_fix+0.0015-(sigz_7_5_fine[3]-sigz_7_5_fine[1])/2, text("$kz7_5", font75)))
# plot!(Xline, Z_7_5, color=:steelblue2)
# plot!(annotations = ([X_ax1], sigz_10_fine[3]+Z_fix+0.0023-(sigz_10_fine[3]-sigz_10_fine[1])/2, text("$kz_10", font10)))
# plot!(Xline, Z_10, color=:dodgerblue)
# plot!(annotations = ([X_ax1], sigz_12_5_fine[3]+Z_fix+0.001-(sigz_12_5_fine[3]-sigz_12_5_fine[1])/2, text("$kz12_5", font125)))
# plot!(Xline, Z_12_5, color=:dodgerblue3)
# # plot!(annotations = ([X_ax2], [Y_ax1], "1", 9))
# plot!(annotations = ([Y_ax], 0.49,text("{\\sigma_z/D}", fontylabel)))
# plot!(annotations = ([X_ax], sigz_12_5[end]-0.005,text("12.5°", font125_c)))
# plot!(annotations = ([X_ax], sigz_10[end], text("10.0°", font10_c)))
# plot!(annotations = ([X_ax], sigz_7_5[end], text("7.5°", font75_c)))
# plot!(annotations = ([X_ax], sigz_5[end], text("5.0°", font5_c)))
# plot!(annotations = ([X_ax], sigz_2_5[end], text("2.5°", font25_c)), left_margin=12Plots.mm, transparent=true, tickfontsize=12, guidefontsize=12)
# savefig("sigma_z_fit_opt_pow.pdf")

# # Rotating annotations
# font25 = text("", 9).font
# font25.rotation=20
# font5 = text("", 9).font
# font5.rotation=22
# font75 = text("", 9).font
# font75.rotation=25
# font10 = text("", 9).font
# font10.rotation=27
# font125 = text("", 9).font
# font125.rotation=27

# plot(slices, sigy_2_5,color=:lightskyblue2, label="2.5°")
# plot!(slice_fine[6:end], sigmaY_2_5[11:end-M], seriestype=:scatter, color=:lightskyblue2, label="2.5°")
# plot!(slice_fine[6:end], sigmaY_5[11:end-M], seriestype=:scatter, color=:skyblue2, label="2.5°")
# plot!(slice_fine[6:end], sigmaY_7_5[11:end-M], seriestype=:scatter, color=:steelblue2, label="2.5°")
# plot!(slice_fine[6:end], sigmaY_10[11:end-M], seriestype=:scatter, color=:dodgerblue, label="2.5°")
# plot!(slice_fine[6:end], sigmaY_12_5[11:end-M], seriestype=:scatter, color=:dodgerblue3, label="2.5°")
# plot!(slices, sigy_5,color=:skyblue2, label="5°")
# plot!(slices, sigy_7_5,color=:steelblue2, label="7.5°")
# plot!(slices, sigy_10,color=:dodgerblue,label="10°")
# plot!(slices, sigy_12_5,color=:dodgerblue3, label="12.5°", xlabel="x/D (Downstream Distance)", xrotation = 0,legend=:false,xlimit = [7, 12.5], legendfontsize=10, grid=false, xticks=[7, 8, 9, 10, 11, 12], dpi=300)
# plot!(annotations = ([X_ax1], sigy_2_5_fine[3]+Z_fix-0.001-(sigy_2_5_fine[3]-sigy_2_5_fine[1])/2, text("$ky2_5", font25)))
# plot!(Xline, Y_2_5, color=:lightskyblue2)
# plot!(annotations = ([X_ax1], sigy_5_fine[3]+Z_fix-0.001-(sigy_5_fine[3]-sigy_5_fine[1])/2, text("$ky_5", font5)))
# plot!(Xline, Y_5, color=:skyblue2)
# plot!(annotations = ([X_ax1], sigy_7_5_fine[3]+Z_fix-0.001-(sigy_7_5_fine[3]-sigy_7_5_fine[1])/2, text("$ky7_5", font75)))
# plot!(Xline, Y_7_5, color=:steelblue2)
# plot!(annotations = ([X_ax1], sigy_10_fine[3]+Z_fix+0.0015-(sigy_10_fine[3]-sigy_10_fine[1])/2, text("$ky_10", font10)))
# plot!(Xline, Y_10, color=:dodgerblue)
# plot!(annotations = ([X_ax1], sigy_12_5_fine[3]+Z_fix+0.002-(sigy_12_5_fine[3]-sigy_12_5_fine[1])/2, text("$ky12_5", font125)))
# plot!(Xline, Y_12_5, color=:dodgerblue3)
# plot!(annotations = ([Y_ax], 0.5,text("{\\sigma_y/D}", fontylabel)))
# plot!(annotations = ([X_ax], sigy_2_5[end],text("2.5°", font25_c)))
# plot!(annotations = ([X_ax], sigy_5[end], text("5.0°", font5_c)))
# plot!(annotations = ([X_ax], sigy_7_5[end], text("7.5°", font75_c)))
# plot!(annotations = ([X_ax], sigy_10[end], text("10.0°", font10_c)))
# plot!(annotations = ([X_ax], sigy_12_5[end], text("12.5°", font125_c)), left_margin=12Plots.mm, transparent=true, tickfontsize=12, guidefontsize=12)
# savefig("sigma_y_fit_opt_pow.pdf")



# # Hold Ct to be constant (but this doesn't seem right)
# # https://www.mdpi.com/2075-1702/7/1/15
# # Ct is a function of yaw, observed experimentally
# # So Ct must be a function of tilt. How to find Ct with the SOWFA data?

# # Ct must be constrained to 0 to 1
# # think about other constraints



# check_with_original(x)













# # Gompertz distribution
# Gompertz_dist(x, η, b) = @. (b*η)*exp(η + (b*x) - (η*exp(10.0*b*x)))

# """SOWFA Single Turbine Filenames"""
# # file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_001_sp7_1turb_hNormal_D126_tilt_-15/lite_data_001.ftr"
# # file_1 = "/Users/jamescutler/Downloads/çbyu_tilt_runs_single_opp/c_002_sp7_1turb_hNormal_D126_tilt_-20/lite_data_002.ftr"
# # file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_003_sp7_1turb_hNormal_D126_tilt_-35/lite_data_003.ftr"

# """SOWFA Single Turbine Filenames"""
# # file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single/c_000_sp7_1turb_hNormal_D126_tilt_base/lite_data_000.ftr"
# # file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_001_sp7_1turb_hNormal_D126_tilt_2.5/lite_data_001.ftr"
# # file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single/c_001_sp7_1turb_hNormal_D126_tilt_5/lite_data_001.ftr"
# # file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_002_sp7_1turb_hNormal_D126_tilt_7.5/lite_data_002.ftr"
# # file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single/c_002_sp7_1turb_hNormal_D126_tilt_10/lite_data_002.ftr"
# # file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_003_sp7_1turb_hNormal_D126_tilt_12.5/lite_data_003.ftr"

# # for power gains
# file_0 = "/Users/jamescutler/Downloads/byu_tilt_runs_single/c_000_sp7_1turb_hNormal_D126_tilt_base/lite_data_000.ftr"
# file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_001_sp7_1turb_hNormal_D126_tilt_2.5/lite_data_001.ftr"
# file_2 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_004_sp7_1turb_hNormal_D126_tilt_5/lite_data_004.ftr"
# file_3 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_002_sp7_1turb_hNormal_D126_tilt_7.5/lite_data_002.ftr"
# file_4 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_005_sp7_1turb_hNormal_D126_tilt_10/lite_data_005.ftr"
# file_5 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_003_sp7_1turb_hNormal_D126_tilt_12.5/lite_data_003.ftr"

# # for power gains
# filename0 = "analysis_smoothing_piece_wise_largeest_span_base_normal_n.jld"
# filename1 = "analysis_smoothing_piece_wise_largeest_span_2.5_normal_n.jld"
# filename2 = "analysis_smoothing_piece_wise_largeest_span_5_normal_n.jld"
# filename3 = "analysis_smoothing_piece_wise_largeest_span_7.5_normal_n.jld"
# filename4 = "analysis_smoothing_piece_wise_largeest_span_10_normal_n.jld"
# filename5 = "analysis_smoothing_piece_wise_largeest_span_12.5_normal_n.jld"

# # y and z array
# y_array = 2245.0:0.5:2745.0
# z_array = 5.0:((304.99999999999994-5.0)/1000):304.99999999999994

# # Final data values
# # filename0 = "analysis_smoothing_piece_wise_largeest_span_base_normal_n_final.jld"
# # filename1 = "analysis_smoothing_piece_wise_largeest_span_2.5_normal_n_final.jld"
# # filename2 = "analysis_smoothing_piece_wise_largeest_span_5_normal_n_final.jld"
# # filename3 = "analysis_smoothing_piece_wise_largeest_span_7.5_normal_n_final.jld"
# # filename4 = "analysis_smoothing_piece_wise_largeest_span_10_normal_n_final.jld"
# # filename5 = "analysis_smoothing_piece_wise_largeest_span_12.5_normal_n_final.jld"


# # filename1 = "analysis_25.jld"
# # filename2 = "analysis_neg15.jld"
# # filename1 = "analysis_smoothing_piece_wise_largeest_span_25_normal.jld"


# # files = [file_1]
# # movnames = ["25"]
# # filenames = [filename1]

# # # Final Data info
# # files = [file_1, file_2, file_3, file_4, file_5]
# # movnames = ["2.5", "5", "7.5", "10", "12.5"]
# # filenames = [filename1, filename2, filename3, filename4, filename5]
# # tilt = [2.5, 5.0, 7.5, 10.0, 12.5] * pi/180.0

# # files = [file_0, file_1, file_2, file_3, file_4, file_5]
# # movnames = ["neg5", "2.5", "5", "7.5", "10", "12.5"]
# # filenames = [filename0, filename1, filename2, filename3, filename4, filename5]
# # tilt = [-5.0, 2.5, 5.0, 7.5, 10.0, 12.5] * pi/180.0

# # Power gain info
# files = [file_1, file_2, file_3, file_4, file_5]
# movnames = ["2.5", "5", "7.5", "10", "12.5"]
# filenames = [filename1, filename2, filename3, filename4, filename5]
# tilt = [2.5, 5.0, 7.5, 10.0, 12.5] * pi/180.0
# slices = 7.0:1.0:12.0
# Uinf = 8.1      # m/s

# params = params_struct(Uinf, slices, tilt, filename1, filename2, filename3, filename4, filename5)

# # generate objective function wrapper
# obj_func!(g,x) = objective_vel_wrapper!(g, x, params)

# # initialize design variable Vector
# CTA = 27.2
# CTB = -0.78
# CTC = 10.5
# CTb = 0.4
# CTη = 0.008
# kz1 = -0.563
# kz2 = 0.108
# kz3 = 0.027
# ky1 = 0.048
# ky2 = 0.018
# sigy01 = 0.266
# sigz01 = 0.168
# sigz02 = -0.014
# sigz03 = 0.0419
# x0 = [CTA, CTB, CTC, CTη, CTb, kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03]

# # IPOPT options
# ip_options = Dict("max_iter" => 50, "tol" => 1e-6)
# solver = IPOPT(ip_options)

# options = Options(solver=solver, derivatives=ForwardAD())


# ng = 30
# lx = [-Inf, -Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf, -Inf,0.01]
# ux = [Inf, Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf, Inf,Inf]
# # lg=[-Inf*ones(ng)]
# # ug=[zeros(ng)]
# lg=-Inf
# ug=0


# # optimize
# x0_initial = x0
# iterations = 20

# filename_opt = "optimal_solution.jld"
# for i = 1:iterations
#     if i == 1
#         fopt_optimal = 1000
#         best = x0
#     end
#     xopt, fopt, info, out = minimize(obj_func!, x0, ng, lx, ux, lg, ug, options)
#     if fopt < fopt_optimal
#         fopt_optimal = fopt
#         best = xopt
#         save(filename_opt, "xopt", best, "fopt", fopt_optimal)
#     end
# end