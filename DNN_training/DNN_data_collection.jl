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
# using ForwardDiff
# using SNOW
using LaTeXStrings
using Images
using FileIO


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
    # savefig("gauss_fit_check.png")

    plot(X_plot, Gfit_plot)
    # savefig("gaussfit_check2.png")
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
    # savefig("gauss_fit_checky.png")

    plot(X_plot, Gfit_plot)
    # savefig("gaussfit_checky2.png")
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
            slice_x_d = slices[i]
            print("SLICE: ", slices[i], "\n")
            slice = gridpoint((T_pos + slices[i]*D), 10.0)
            moviename = join([movname, "_", string(slices[i]), "_", smoothing[j], "_n_no_legend", ".pdf"])
            DNNname = join([movname, "_", string(slices[i]), "_DNN", ".jpg"])
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
            # savefig(gauss_name_z)
            

            # plotting full gaussain fit for upper profile fit
            plot(crop_Z_plot_fine/90, Vert_prof_previous,linewidth=linew, color=:black,linestyle=:dot, label="SOWFA")
            plot!(Z_val_upper/90, Gfit_array_Z_upper_plot_full,linewidth=5, color=:dodgerblue3, label="Gaussian Fit", xlimit=[0,3.5], grid=:false, xlabel="z*", ylabel="Velocity Deficit (m/s)", xrotation = 0,foreground_color_legend = nothing,yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz)
            # savefig(gauss_name_z_full)

            plot((crop_Y_plot_fine.-2500)/126, Horz_profile_fine,linewidth=linew, color=:black,linestyle=:dot, label="SOWFA")
            plot!((Y_val.-2500)/126, Gfit_array_Y,linewidth=5, label="Gaussian Fit", color=:dodgerblue3, grid=:false, xlabel="y/D", ylabel="Velocity Deficit (m/s)", xrotation = 0,foreground_color_legend = nothing,yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz)
            # savefig(gauss_name_y)

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
            power_downstream, avg_vel = power_estimate(tilt, slice_x_d, cropped_SOWFA_data_comp_fine, 90.0, 2500.0, D, crop_Z_plot_fine, crop_Y_plot_fine, power_filename)
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
            # savefig(moviename)

            # scale the matrix to [0, 255]
            matrix = cropped_SOWFA_data_comp_neg_fine
            image_scaled = round.(Int, (255/8.1)*(matrix))

            img = Gray.(image_scaled)
            save(DNNname, img)
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
    slice_x_d = slice
    print("SLICE: ", string(slice), "\n")
    str_slice = string(slice)
    slice = gridpoint((T_pos + slice*D), 10.0)
    DNNname = join([movname, "_", str_slice, "_DNN", ".jpg"])
    print("DNNname: ", DNNname, "\n")
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


    # finestep = ((maximum(crop_Z_plot_comp)-minimum(crop_Z_plot_comp))/finenum)
    # """Test to find all peaks in slice"""
    # """If there are multiple peaks then use Threshold hybrid method"""
    # """Otherwise it means we are significantly downstream that we can just find maximum velocity-deficit"""
    # """Or if we are significantly downstream (dist>7D) and there are multiple peaks it means there is a kidney bean shape"""

    # """Find maximum points in wake"""
    # Global_max = findmax(cropped_SOWFA_data_comp_neg_fine)
    # Y_global_max = crop_Y_plot_fine[Global_max[2][2]]
    # Z_global_max = crop_Z_plot_fine[Global_max[2][1]]
    # print("center_Y: ", Y_global_max, "\n")
    # print("center_Z: ", Z_global_max)
    # Global_Horz_profile = cropped_SOWFA_data_comp_neg_fine[Global_max[2][1],:]
    # Global_Vert_profile = cropped_SOWFA_data_comp_neg_fine[:,Global_max[2][2]]
    

    # """Find maximum points in original data"""


    # """Trim Profile for gaussian fit"""
    # # anything in profiles above 0.1 helps get rid of edges
    # Trim_edges_horz = findall(t -> t<0.8, Global_Horz_profile)
    # Trim_edges_vert = findall(t -> t<0.8, Global_Vert_profile)

    # # Remove edges of velocity profiles for more accurate fit
    # # deleteat!(Global_Horz_profile, Trim_edges_horz)
    # # deleteat!(Global_Vert_profile, Trim_edges_vert)
    # crop_Z_plot_fine = convert(Array{Float64,1}, crop_Z_plot_fine)
    # crop_Y_plot_fine = convert(Array{Float64,1}, crop_Y_plot_fine)
    # # deleteat!(crop_Z_plot_fine, Trim_edges_vert)
    # # deleteat!(crop_Y_plot_fine, Trim_edges_horz)

    # Horz_profile_fine = Global_Horz_profile

    # """Do Gaussian Fit"""
    # # use trimmed profiles, no need to find max, just use max found earlier

    # # Prepare mirrored gaussian profiles to better match portion of wake that will be
    # # running into downstream turbines

    # # Using moving average to smooth out Vertical Profile
    # Vert_prof_previous = Global_Vert_profile
    # Vert_prof_previous_fine = Global_Vert_profile
    # Vert_profile_fine = Global_Vert_profile
    # print(size(crop_Z_plot_fine), "\n")
    # print(size(Vert_profile_fine), "\n")

    # # Interpolate Vert_profile first (an initial smoothing)
    # interp_Z_spline = CubicSpline(crop_Z_plot_fine, Vert_profile_fine)
    # ZZ = minimum(crop_Z_plot_fine):0.005:maximum(crop_Z_plot_fine)
    # interp_Z_profile = interp_Z_spline(ZZ)

    # ZZ_smooth = ZZ
    # # interp_Z_profile = movingaverage(interp_Z_profile, avg_factor)

    # # # This smoothes out spikes effeciently, but it can also unrealistically lower peak velocity deficit

    # Max_loc_Z = findmax(interp_Z_profile)
    # Z_cm_comp = ZZ[Max_loc_Z[2]]

    # """Upper Profile"""
    # ZZ_upper = ZZ[Max_loc_Z[2]:end]
    # Upper_Z_profile = interp_Z_profile[Max_loc_Z[2]:end]

    # # flip it and append it to make a whole velocity profile
    # Lower_Z_profile = reverse(Upper_Z_profile)
    # Interp_Z_profile = vcat(Lower_Z_profile, Upper_Z_profile)

    # # Adjust ZZ to match Interp_Z_profile
    # # create array of acending values from 1 to length(ZZ_upper)
    # Adjust_array = 0.005:0.01:length(Lower_Z_profile)*0.01
    # ZZ_lower = reverse(ZZ_upper.-Adjust_array)
    # ZZ = vcat(ZZ_lower, ZZ_upper)
    # plot_upper = findmax(Interp_Z_profile)

    # # for plotting 
    # Interp_Z_upper_plot = Interp_Z_profile[plot_upper[2]:end]
    # ZZ_upper_plot = ZZ[plot_upper[2]:end]

    # # now take gaussian fit of it.
    # # print("size ZZ: ", size(ZZ), "\n")
    # # global parameters = Param_val(ZZ[12870:end-12870], Interp_Z_profile[12870:end-12870])
    # # global parameters = Param_val(ZZ, Interp_Z_profile)
    # # Gfit_vert = fit(Normal, ZZ_ground, Interp_Z_profile_ground)
    # # Gfit_vert = fit_gauss(parameters)       # manual fit
    # # Z_val_upper = -200:0.01:300
    # # Gfit_array_Z_upper = gaussvalues(Gfit_vert["σ"], Gfit_vert["μ"], Gfit_vert["A"], Z_val_upper)
    # # sigma_z_d_comp = (Gfit_vert["σ"])/D
    

    # # gaussian fit
    # # Gfit_vert = fit(Normal, crop_Z_plot, Vert_profile)
    # Gfit_vert = fit(Normal, ZZ, Interp_Z_profile)
    # Z_val_upper = -200:0.01:300
    # Gfit_array_Z_upper = pdf(Gfit_vert, Z_val_upper)
    # sigma_z_d_comp = (Gfit_vert.σ)/D
    # # print("upper: ", Gfit_vert.μ, "\n")


    # addition = 0.0
    # # solve for constant to multiply gfit_array_Z_upper by 
    # max_c = findmax(Gfit_array_Z_upper)
    # Max_def_z = Gfit_array_Z_upper[max_c[2]]
    # multiply_c = (Max_loc_Z[1]+addition)/Max_def_z

    # # Gfit_array_Z_upper = Gfit_array_Z_upper*Gfit_vert["A"]
    # Gfit_array_Z_upper = Gfit_array_Z_upper
    # Gfit_array_Z_upper_plot = Gfit_array_Z_upper[max_c[2]:end]*multiply_c
    # Gfit_array_Z_upper_plot_full = Gfit_array_Z_upper*multiply_c
    # Z_val_upper_plot = Z_val_upper[max_c[2]:end]


    # """Lower Profile"""
    # ZZ_lower = ZZ_smooth[1:Max_loc_Z[2]]
    # Lower_Z_profile = interp_Z_profile[1:Max_loc_Z[2]]
    # Upper_Z_profile = reverse(Lower_Z_profile)
    # Interp_Z_profile_ground = vcat(Lower_Z_profile, Upper_Z_profile)

    # # flip it and append it to make a whole velocity profile
    # Adjust_array = 0.005:0.01:length(Lower_Z_profile)*0.01
    # ZZ_higher = reverse(ZZ_lower.+reverse(Adjust_array))
    # ZZ_ground = vcat(ZZ_lower, ZZ_higher)

    # # for plotting
    # plot_under = findmax(Interp_Z_profile_ground)
    # Interp_Z_lower_plot = Interp_Z_profile_ground[1:plot_under[2]]
    # ZZ_under_plot = ZZ_ground[1:plot_under[2]]

    # """THIS FIT ISN'T PERFORMING WELL"""
    # """NEED TO WRITE MANUAL SOLVE FOR SIGMA"""

    
    
    # # global parameters = Param_val(ZZ_ground[12870:end-12870], Interp_Z_profile_ground[12870:end-12870])
    # global parameters = Param_val(ZZ_ground, Interp_Z_profile_ground)
    # Gfit_vert = fit(Normal, ZZ_ground, Interp_Z_profile_ground)
    # # Gfit_vert = fit_gauss(parameters)       # manual fit
    
    # Z_val_ground = -200:0.01:300
    # Gfit_array_Z_ground = pdf(Gfit_vert, Z_val_ground)
    # sigma_z_d_comp_ground = (Gfit_vert.σ)/D

    # # Z_val_ground = -200:0.01:300
    # # Gfit_array_Z_ground = gaussvalues(Gfit_vert["σ"], Gfit_vert["μ"], Gfit_vert["A"], Z_val_ground)
    # # sigma_z_d_comp_ground = (Gfit_vert["σ"])/D
    

    # # solve for constant to multiply gfit_array_Z_ground by 
    # max_c = findmax(Gfit_array_Z_ground)
    # Max_def_z = Gfit_array_Z_ground[max_c[2]]
    # multiply_c_ground = (Max_loc_Z[1]+addition)/Max_def_z

    # # Gfit_array_Z_ground = Gfit_array_Z_ground*Gfit_vert["A"]
    # Gfit_array_Z_ground = Gfit_array_Z_ground
    # Gfit_array_Z_ground_plot = Gfit_array_Z_ground[1:max_c[2]]*multiply_c_ground
    # Z_val_ground_plot = Z_val_ground[1:max_c[2]]

    # """Horizontal"""
    # Gfit_horz = fit(Normal, crop_Y_plot_fine, Horz_profile_fine)
    # Y_val = 2250:0.005:2750
    # Max_loc_Y = findmax(Horz_profile_fine)
    # Gfit_array_Y = pdf(Gfit_horz, Y_val)
    # max_cy = findmax(Gfit_array_Y)
    # Max_def_y = Gfit_array_Y[max_cy[2]]
    # multiply_c_y = Max_loc_Y[1]/Max_def_y
    # maxy = findmax(Gfit_array_Y)
    # Y_cm_comp = Y_val[maxy[2]]
    # sigma_y_d_comp = (Gfit_horz.σ)/D

    # Gfit_array_Y = Gfit_array_Y*multiply_c_y


    # fontsz = 13
    # linew = 2
    # marks = 5



    # # sigmay[i,j] = sigma_y_d_comp
    # # sigmaz[i,j] = sigma_z_d_comp
    # # sigmaz_ground[i,j] = sigma_z_d_comp_ground
    # # defz[i,j] = Z_cm_comp
    # # defy[i,j] = Y_cm_comp
    # # cup[i,j] = multiply_c
    # # cdown[i,j] = multiply_c_ground
    # power_downstream, avg_vel = power_estimate(tilt, slice_x_d, cropped_SOWFA_data_comp_fine, 90.0, 2500.0, D, crop_Z_plot_fine, crop_Y_plot_fine, power_filename)
    # # power_upstream = power_estimate_tilt(cropped_SOWFA_data_ambient_fine,90.0, 2500.0, D, crop_Z_plot_fine, crop_Y_plot_fine, tilt, 1.88, power_filename_down)
    # # power[i,j] = power_downstream
    # # power_up[i,j] = power_upstream
    # # avg_V[i,j] = avg_vel


    
    # data1 = heatmap((crop_Y_plot_fine.-2500)/126, (crop_Z_plot_fine/90),cropped_SOWFA_data_comp_neg_fine, c=:Blues)
    # # plot(data1, colorbar_title=" \nVelocity Deficit (m/s)",clim=(0,1.35), right_margin=5Plots.mm, legend=:false, grid=:false, xlabel="y/D", ylabel="z*", xrotation = 0)
    # plot(data1, colorbar_title=" \nVelocity Deficit (m/s)",clim=(0,1.2), right_margin=5Plots.mm, grid=:false, xlabel="y/D", ylabel="z*")
    # # plot(data1, colorbar_title=" \nVelocity Deficit (m/s)",clim=(0,0.75), right_margin=5Plots.mm, legend=:false, grid=:false, xlabel="y/D", ylabel="z*", xrotation = 0)

    # # plot(data1, colorbar_title="Velocity Deficit (m/s)",clim=(0,0.75), legend=:false, grid=:false, xlabel="y/D", ylabel="z*", xrotation = 0)
    # # # plot!([crop_Y_plot[min_loc[1]]], [crop_Z_plot[min_loc[2]]], color="blue")
    # # # scatter!([crop_Y_plot[min_loc[2]]], [crop_Z_plot[min_loc[1]]], color="blue", label="wake center")
    # f(t) = ((126/2).*cos.(t))/126
    # g(t) = (90 .+ (126/2).*sin.(t))/90
    # range = 0:0.01:2*pi
    # x_values = f(range)
    # y_values = g(range)

    # # # scatter!((crop_Y_plot_comp[Y_whole_wake_comp].-2500)/126, (crop_Z_plot_comp[X_whole_wake_comp])/90, color="red", label="Wake")
    # scatter!([0.0], [1.0],markerstrokewidth=0, markersize = marks, color="firebrick", label="")
    # # # scatter!(([Y_cm_comp_threshold].-2500)/126, [Z_cm_comp_threshold]/90, color="brown", label="Center based on max Deficit")
    # # # scatter!(([Y_cm_comp_max].-2500)/126, [Z_cm_comp_max]/90, color="magenta", label="Center based on max Deficit")
    # scatter!(([Y_cm_comp].-2500)/126, [Z_cm_comp]/90,markerstrokewidth=0, markersize = marks, color="green", label="Wake Center")
    # plot!(x_values, y_values, label="Rotor",linewidth=linew+1,legend=:false, color="firebrick", xlabel="y/D", ylabel="z*", xrotation = 0)

    # xlims!((-2,2))
    # ylims!((0,3.388888889))
    # # plot!(size=((14/8)*400,400),yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz,foreground_color_legend = nothing,)

    # plot!(size=((14/8)*400,400),yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz,foreground_color_legend = nothing)
    # savefig(moviename)
    # data1 = heatmap((crop_Y_plot_fine.-2500)/126, (crop_Z_plot_fine/90),cropped_SOWFA_data_comp_neg_fine, c=:Blues)
    # data1 = heatmap(cropped_SOWFA_data_comp_neg_fine, c=:Blues)
    # plot(data1)
    # savefig(moviename)

    matrix = cropped_SOWFA_data_comp_neg_fine
    image_scaled = (1.0/8.1)*(matrix)

    img = Gray.(image_scaled)
    save(DNNname, img)

    return cropped_SOWFA_data_comp_neg, 0
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

function power_estimate(tilt, slice, velocity_slice, hub_height, Y, D, Z_array, Y_array, power_filename)
    counter = 0
    Velocity_Total = 0
    # i represents Z ?
    # j represents Y?
    # print("length(velocity_slice[:,1]: ", length(velocity_slice[:,1]), "\n")
    # print("length(velocity_slice[:,1]: ", length(velocity_slice[1,:]), "\n")
    # print("Z_array: ", length(Z_array), "\n")
    # print("Y_array: ", length(Y_array), "\n")

    # Adjust hub_height to include deflection
    print("tilt: ", tilt, "\n")
    print("slice: ", slice, "\n")
    def_z = 90.0.+deflection_est(tilt, slice)*126.0
    print("def_z: ", def_z, "\n")
    print("hun_height: ", hub_height, "\n")

    hub_height = def_z

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
            
            if distance < D/6

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
    # savefig(wakesave)
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

function SOWFA_multiple_save(movnames, files, filenames, tilt)
    for j in 1:length(files)
        file_1 = files[j]
        filename = filenames[j]
        movnam = movnames[j]

        # slice = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        D = 126
        # slice = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        slices = 1:0.1:21.3

        # What are these for the original SOWFA data?
        z_array = 31
        y_array = 51
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

        # Include something here where we compute Ct
        # V_inf = 8.1     # m/s
        # V_w = avgV_val       # m/S
        # V_d = @. (0.5)*(V_inf + V_w)       # velocity at rotor
        # CT = @. (V_d*(V_inf - V_w))/(0.5*V_inf^2)       # Ct based on multiple downstream distances

        # a = @. ((V_w/V_inf)-1)/(-2)
        # CT = @. 4*a*(1-a)
        # CT_alt = @. 4*a*sqrt(1 - a*(2*cos(tilt) - a))


        # save(filename, "Vel_slices", Slices_vel, "avgV", avgV_val, "Ct", CT, "Ct_alt", CT_alt)
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

        # save(filename, "sigy", sigmay_ht, "sigz", sigmaz_ht, "defz", defz_ht, "defy", defy_ht, "sigz_g", sigmaz_g, "cup", cup, "cdown", cdown, "power", power_t, "power_up", power_up_val, "avgV", avgV_t)
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

"""SOWFA Single Turbine Filenames"""
# file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_001_sp7_1turb_hNormal_D126_tilt_-15/lite_data_001.ftr"
# file_1 = "/Users/jamescutler/Downloads/çbyu_tilt_runs_single_opp/c_002_sp7_1turb_hNormal_D126_tilt_-20/lite_data_002.ftr"
# file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_003_sp7_1turb_hNormal_D126_tilt_-35/lite_data_003.ftr"

"""SOWFA Single Turbine Filenames"""
# file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single/c_000_sp7_1turb_hNormal_D126_tilt_base/lite_data_000.ftr"
# file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_001_sp7_1turb_hNormal_D126_tilt_2.5/lite_data_001.ftr"
# file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single/c_001_sp7_1turb_hNormal_D126_tilt_5/lite_data_001.ftr"
# file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_002_sp7_1turb_hNormal_D126_tilt_7.5/lite_data_002.ftr"
# file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single/c_002_sp7_1turb_hNormal_D126_tilt_10/lite_data_002.ftr"
# file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_003_sp7_1turb_hNormal_D126_tilt_12.5/lite_data_003.ftr"

# for power gains
file_0 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_003_sp7_1turb_hNormal_D126_tilt_-35/lite_data_003.ftr"
file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_002_sp7_1turb_hNormal_D126_tilt_-20/lite_data_002.ftr"
file_2 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_001_sp7_1turb_hNormal_D126_tilt_-15/lite_data_001.ftr"
file_3 = "/Users/jamescutler/Downloads/byu_tilt_runs_single/c_000_sp7_1turb_hNormal_D126_tilt_base/lite_data_000.ftr"
file_4 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_001_sp7_1turb_hNormal_D126_tilt_2.5/lite_data_001.ftr"
file_5 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_004_sp7_1turb_hNormal_D126_tilt_5/lite_data_004.ftr"
file_6 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_002_sp7_1turb_hNormal_D126_tilt_7.5/lite_data_002.ftr"
file_7 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_005_sp7_1turb_hNormal_D126_tilt_10/lite_data_005.ftr"
file_8 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_003_sp7_1turb_hNormal_D126_tilt_12.5/lite_data_003.ftr"




# final Data Values
# file_0 = "/Users/jamescutler/Downloads/byu_tilt_runs_single/c_000_sp7_1turb_hNormal_D126_tilt_base/lite_data_000.ftr"
# file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_001_sp7_1turb_hNormal_D126_tilt_2.5/lite_data_001.ftr"
# file_2 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_004_sp7_1turb_hNormal_D126_tilt_5/lite_data_004.ftr"
# file_3 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_002_sp7_1turb_hNormal_D126_tilt_7.5/lite_data_002.ftr"
# file_4 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_005_sp7_1turb_hNormal_D126_tilt_10/lite_data_005.ftr"
# file_5 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_003_sp7_1turb_hNormal_D126_tilt_12.5/lite_data_003.ftr"



# file_1 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_003_sp7_1turb_hNormal_D126_tilt_25/lite_data_003.ftr"
# file_2 = "/Users/jamescutler/Downloads/byu_tilt_runs_single_opp/c_001_sp7_1turb_hNormal_D126_tilt_-15/lite_data_001.ftr"



# filename1 = "analysis_smoothing_piece_wise_largeest_span_2.5_no_smooth.jld"
# filename2 = "analysis_smoothing_piece_wise_largeest_span_5_no_smooth.jld"
# filename3 = "analysis_smoothing_piece_wise_largeest_span_7.5_no_smooth.jld"
# filename4 = "analysis_smoothing_piece_wise_largeest_span_10_no_smooth.jld"
# filename5 = "analysis_smoothing_piece_wise_largeest_span_12.5_no_smooth.jld"

# filename1 = "analysis_smoothing_piece_wise_largeest_span_2.5.jld"
# filename2 = "analysis_smoothing_piece_wise_largeest_span_5.jld"
# filename3 = "analysis_smoothing_piece_wise_largeest_span_7.5.jld"
# filename4 = "analysis_smoothing_piece_wise_largeest_span_10.jld"
# filename5 = "analysis_smoothing_piece_wise_largeest_span_12.5.jld"

# for power gains
# filename0 = "analysis_smoothing_piece_wise_largeest_span_base_normal_n.jld"
# filename1 = "analysis_smoothing_piece_wise_largeest_span_2.5_normal_n.jld"
# filename2 = "analysis_smoothing_piece_wise_largeest_span_5_normal_n.jld"
# filename3 = "analysis_smoothing_piece_wise_largeest_span_7.5_normal_n.jld"
# filename4 = "analysis_smoothing_piece_wise_largeest_span_10_normal_n.jld"
# filename5 = "analysis_smoothing_piece_wise_largeest_span_12.5_normal_n.jld"


filename0 = "analysis_smoothing_piece_wise_largeest_span_-35_normal_originalSOWFA.jld"
filename1 = "analysis_smoothing_piece_wise_largeest_span_-20_normal_originalSOWFA.jld"
filename2 = "analysis_smoothing_piece_wise_largeest_span_-15_normal_originalSOWFA.jld"
filename3 = "analysis_smoothing_piece_wise_largeest_span_base_normal_originalSOWFA.jld"
filename4 = "analysis_smoothing_piece_wise_largeest_span_2.5_normal_originalSOWFA.jld"
filename5 = "analysis_smoothing_piece_wise_largeest_span_5_normal_originalSOWFA.jld"
filename6 = "analysis_smoothing_piece_wise_largeest_span_7.5_normal_originalSOWFA.jld"
filename7 = "analysis_smoothing_piece_wise_largeest_span_10_normal_originalSOWFA.jld"
filename8 = "analysis_smoothing_piece_wise_largeest_span_12.5_normal_originalSOWFA.jld"


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

# Power gain info
files = [file_8]
movnames = ["12.5"]
filenames = [filename8]
tilt = [12.5] * pi/180.0

# files = [file_2]
# movnames = ["neg15"]
# filenames = [filename2]

# files = [file_5]
# movnames = ["12.5"]
# filenames = [filename5]

# files = [file_1]
# movnames = ["25"]
# filenames = [filename1]

# files = [file_1]
# movnames = ["2.5"]
# filenames = [filename1]

tilt = [12.5] * pi/180.0
SOWFA_multiple_save(movnames, files, filenames, tilt)
