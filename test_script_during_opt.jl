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

Gompertz_dist(x, η, b) = @. (b*η)*exp(η + (b*x) - (η*exp(10.0*b*x)))
# run_optimization()
filename_opt = "optimal_solution_newc.jld"
x = load(filename_opt, "xopt")
fopt = load(filename_opt, "fopt")
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

A = 27.2
B = -0.78
C = 10.5
C_T_new_adjusted = 7.620929940139257175e-01 .+ 1A*Gompertz_dist((tilt*C).+B, 0.008, 0.4)

tilt = [2.5, 5.0, 7.5, 10.0, 12.5]*pi/180.0
# check CT
C_T_opt = CTA*Gompertz_dist((tilt*CTC).+CTB, CTη, CTb)
plot(tilt, C_T_new_adjusted)
plot!(tilt, C_T_opt)

# check kz
kz_pred = @. -0.563*(tilt^2) + 0.108*tilt + 0.027
kz_opt = @. kz1*(tilt^2) + kz2*tilt + kz3
plot(tilt*(180/pi), kz_pred, label="pred")
plot!(tilt*(180/pi), kz_opt, label="opt")

# check ky
ky_pred = @. 0.048*tilt + 0.018
ky_opt = @. ky1*tilt + ky2
plot(tilt*(180/pi), ky_pred, label="pred")
plot!(tilt*(180/pi), ky_opt, label="opt")
# check sigy0


# check sigz0
sigz_pred = @. 0.168 - 0.014*log(tilt - 0.0419)
sigz0_opt = @. sigz01 + sigz02*log(tilt + sigz03)
plot(tilt*(180/pi), sigz_pred, label="pred")
plot!(tilt*(180/pi), sigz0_opt, label="opt")

slices = 7:1.0:12.0

sigz_2_5, sigy_2_5 = sigysigz_solve(slices, tilt[1], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
sigz_5, sigy_5 = sigysigz_solve(slices, tilt[2], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
sigz_7_5, sigy_7_5 = sigysigz_solve(slices, tilt[3], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
sigz_10, sigy_10 = sigysigz_solve(slices, tilt[4], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)
sigz_12_5, sigy_12_5 = sigysigz_solve(slices, tilt[5], kz1, kz2, kz3, ky1, ky2, sigy01, sigz01, sigz02, sigz03)

plot(slices, sigz_2_5, label="2.5")
plot!(slices, sigz_5, label="5")
plot!(slices, sigz_7_5, label="7.5")
plot!(slices, sigz_10, label="10")
plot!(slices, sigz_12_5, label="12.5")

plot(slices, sigy_2_5, label="2.5")
plot!(slices, sigy_5, label="5")
plot!(slices, sigy_7_5, label="7.5")
plot!(slices, sigy_10, label="10")
plot!(slices, sigy_12_5, label="12.5")
"""To DO"""
# Ct must be constrained to 0 to 1
# think about other constraints

















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