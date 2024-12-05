using NPZ
using Arrow
using DataFrames
using CSV
using Dierckx
# using Plots
using PyPlot
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

# pyplot()
# default(titlefont= ("times"), guidefont=("times"), tickfont=("times"))
PyPlot.matplotlib[:rc]("font", family="serif", serif="Times New Roman")

# Set the font to Times New Roman for LaTeX-style text as well
PyPlot.matplotlib[:rc]("mathtext", fontset="custom")
PyPlot.matplotlib[:rc]("mathtext", rm="Times New Roman")
PyPlot.matplotlib[:rc]("mathtext", it="Times New Roman:italic")
PyPlot.matplotlib[:rc]("mathtext", bf="Times New Roman:bold")


""" Combined Plot of DNN Runs """

s1 = npzread("source_image_tilt_2.5_slice_7.0.npy")
s2 = npzread("source_image_tilt_2.5_slice_8.0.npy")
s3 = npzread("source_image_tilt_2.5_slice_9.0.npy")
s4 = npzread("source_image_tilt_2.5_slice_10.0.npy")
s5 = npzread("source_image_tilt_2.5_slice_11.0.npy")
s6 = npzread("source_image_tilt_2.5_slice_12.0.npy")
s7 = npzread("source_image_tilt_5.0_slice_7.0.npy")
s8 = npzread("source_image_tilt_5.0_slice_8.0.npy")
s9 = npzread("source_image_tilt_5.0_slice_9.0.npy")
s10 = npzread("source_image_tilt_5.0_slice_10.0.npy")
s11 = npzread("source_image_tilt_5.0_slice_11.0.npy")
s12 = npzread("source_image_tilt_5.0_slice_12.0.npy")
s13 = npzread("source_image_tilt_7.5_slice_7.0.npy")
s14 = npzread("source_image_tilt_7.5_slice_8.0.npy")
s15 = npzread("source_image_tilt_7.5_slice_9.0.npy")
s16 = npzread("source_image_tilt_7.5_slice_10.0.npy")
s17 = npzread("source_image_tilt_7.5_slice_11.0.npy")
s18 = npzread("source_image_tilt_7.5_slice_12.0.npy")
s19 = npzread("source_image_tilt_10.0_slice_7.0.npy")
s20 = npzread("source_image_tilt_10.0_slice_8.0.npy")
s21 = npzread("source_image_tilt_10.0_slice_9.0.npy")
s22 = npzread("source_image_tilt_10.0_slice_10.0.npy")
s23 = npzread("source_image_tilt_10.0_slice_11.0.npy")
s24 = npzread("source_image_tilt_10.0_slice_12.0.npy")
s25 = npzread("source_image_tilt_12.5_slice_7.0.npy")
s26 = npzread("source_image_tilt_12.5_slice_8.0.npy")
s27 = npzread("source_image_tilt_12.5_slice_9.0.npy")
s28 = npzread("source_image_tilt_12.5_slice_10.0.npy")
s29 = npzread("source_image_tilt_12.5_slice_11.0.npy")
s30 = npzread("source_image_tilt_12.5_slice_12.0.npy")

g1 = npzread("generated_image_tilt_2.5_slice_7.0.npy")
g2 = npzread("generated_image_tilt_2.5_slice_8.0.npy")
g3 = npzread("generated_image_tilt_2.5_slice_9.0.npy")
g4 = npzread("generated_image_tilt_2.5_slice_10.0.npy")
g5 = npzread("generated_image_tilt_2.5_slice_11.0.npy")
g6 = npzread("generated_image_tilt_2.5_slice_12.0.npy")
g7 = npzread("generated_image_tilt_5.0_slice_7.0.npy")
g8 = npzread("generated_image_tilt_5.0_slice_8.0.npy")
g9 = npzread("generated_image_tilt_5.0_slice_9.0.npy")
g10 = npzread("generated_image_tilt_5.0_slice_10.0.npy")
g11 = npzread("generated_image_tilt_5.0_slice_11.0.npy")
g12 = npzread("generated_image_tilt_5.0_slice_12.0.npy")
g13 = npzread("generated_image_tilt_7.5_slice_7.0.npy")
g14 = npzread("generated_image_tilt_7.5_slice_8.0.npy")
g15 = npzread("generated_image_tilt_7.5_slice_9.0.npy")
g16 = npzread("generated_image_tilt_7.5_slice_10.0.npy")
g17 = npzread("generated_image_tilt_7.5_slice_11.0.npy")
g18 = npzread("generated_image_tilt_7.5_slice_12.0.npy")
g19 = npzread("generated_image_tilt_10.0_slice_7.0.npy")
g20 = npzread("generated_image_tilt_10.0_slice_8.0.npy")
g21 = npzread("generated_image_tilt_10.0_slice_9.0.npy")
g22 = npzread("generated_image_tilt_10.0_slice_10.0.npy")
g23 = npzread("generated_image_tilt_10.0_slice_11.0.npy")
g24 = npzread("generated_image_tilt_10.0_slice_12.0.npy")
g25 = npzread("generated_image_tilt_12.5_slice_7.0.npy")
g26 = npzread("generated_image_tilt_12.5_slice_8.0.npy")
g27 = npzread("generated_image_tilt_12.5_slice_9.0.npy")
g28 = npzread("generated_image_tilt_12.5_slice_10.0.npy")
g29 = npzread("generated_image_tilt_12.5_slice_11.0.npy")
g30 = npzread("generated_image_tilt_12.5_slice_12.0.npy")

A1 = @. abs(s1 - g1)^2
A2 = @. abs(s2 - g2)^2
A3 = @. abs(s3 - g3)^2
A4 = @. abs(s4 - g4)^2
A5 = @. abs(s5 - g5)^2
A6 = @. abs(s6 - g6)^2
A7 = @. abs(s7 - g7)^2
A8 = @. abs(s8 - g8)^2
A9 = @. abs(s9 - g9)^2
A10 = @. abs(s10 - g10)^2
A11 = @. abs(s11 - g11)^2
A12 = @. abs(s12 - g12)^2
A13 = @. abs(s13 - g13)^2
A14 = @. abs(s14 - g14)^2
A15 = @. abs(s15 - g15)^2
A16 = @. abs(s16 - g16)^2
A17 = @. abs(s17 - g17)^2
A18 = @. abs(s18 - g18)^2
A19 = @. abs(s19 - g19)^2
A20 = @. abs(s20 - g20)^2
A21 = @. abs(s21 - g21)^2
A22 = @. abs(s22 - g22)^2
A23 = @. abs(s23 - g23)^2
A24 = @. abs(s24 - g24)^2
A25 = @. abs(s25 - g25)^2
A26 = @. abs(s26 - g26)^2
A27 = @. abs(s27 - g27)^2
A28 = @. abs(s28 - g28)^2
A29 = @. abs(s29 - g29)^2
A30 = @. abs(s30 - g30)^2

A_total = @. A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10 + 
A11 + A12 + A13 + A14 + A15 + A16 + A17 + A18 + A19 + A20 + 
A21 + A22 + A23 + A24 + A25 + A26 + A27 + A28 + A29 + A30

A_total = A_total[end:-1:1, :]

# Define font size variable
font_size_axis = 20


# Define axis scales
crop_Y_plot_fine = range(2245.0, stop=2745.0, length=51)
crop_Z_plot_fine = range(5.0, stop=304.99999999999994, length=31)
Y_scaled = (crop_Y_plot_fine .- 2500) / 126
Z_scaled = (crop_Z_plot_fine .- 90.0) / 90

fig, ax = plt.subplots()
cax = ax.imshow(A_total, extent=[Y_scaled[1], Y_scaled[end], Z_scaled[1], Z_scaled[end]], aspect=0.9, cmap="Blues", clim=(0.0, 10.0))

# Add colorbar with adjusted size
cbar = fig.colorbar(cax, ax=ax, fraction=0.035, pad=0.04)
# cbar.set_label("\$ |ΔU|^2 \$", fontsize=font_size_axis, fontname="Times New Roman")
cbar.ax.tick_params(labelsize=font_size_axis)  # Set colorbar ticks fontsize

# Set labels and limits
ax.set_xlabel("y/D", fontsize=font_size_axis, fontname="Times New Roman")
# ax.set_ylabel("z*", fontsize=font_size_axis, fontname="Times New Roman")
ax.set_xlim([-2.0, 2])
ax.set_ylim([-1.0, 2.45])

# Set custom ticks
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
ax.tick_params(length=0)
ax.tick_params(axis="both", labelsize=font_size_axis)  # Set axis ticks fontsize

# Hide top and right spines
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)

# Custom annotations
ax.text(-2.6, 0.65, "z*", fontsize=font_size_axis, fontname="Times New Roman")
# ax.text(3.05, 0.75, "\$ |ΔU|^2 \$", fontsize=font_size_axis, fontname="Times New Roman")
ax.text(2.7, 0.75, "\$ |Δ\\hat{U}|^2 \$", fontsize=font_size_axis, fontname="Times New Roman")

fig.tight_layout()

fig.savefig("DNN_total_mse_same.pdf", bbox_inches="tight")

