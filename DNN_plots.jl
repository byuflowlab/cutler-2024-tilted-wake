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


train_accuracy = npzread("train_accuracy_2500-2.npy")
x_acc = npzread("x_acc_2500.npy")
val_acc = npzread("val_acc_2500-2.npy")

train_accuracy_5 = npzread("train_accuracy_500_window_4.npy")
x_acc_5 = npzread("x_acc_500_window_4.npy")
val_acc_5 = npzread("val_acc_500_window_4.npy")

train_accuracy_more = npzread("train_accuracy_test.npy")
x_acc_more = npzread("x_acc_test.npy")
val_acc_more = npzread("val_acc_test.npy")

# Plotting
fig, ax = plt.subplots()

# First plot
# ax.plot(x_acc, train_accuracy, color="black", label="SSIM")

# Second plot
ax.plot(x_acc_5, train_accuracy_5, color="black")

# Third plot
ax.plot(x_acc_more .+ x_acc_5[end], train_accuracy_more, color="dodgerblue")

# Set labels and fonts
ax.set_xlabel("Epoch", fontsize=font_size_axis, fontname="Times New Roman")
ax.set_ylabel("", fontsize=font_size_axis, fontname="Times New Roman")
ax.set_ylim(0, 0.0005)
ax.tick_params(axis="both", which="major", labelsize=14, labelcolor="black", width=1.5)

# Custom annotations
ax.text(-1000.0, 0.00025, "RMS", fontsize=font_size_axis, fontname="Times New Roman")
ax.text(1550.0, 0.00015, "SSIM", fontsize=font_size_axis, fontname="Times New Roman")
ax.text(4100.0, 0.0001, "RMS", fontsize=font_size_axis, fontname="Times New Roman", color="dodgerblue")


# Remove grid and legend if necessary
ax.grid(false)
ax.legend().set_visible(false)
ax.set_xlim([0, x_acc_more[end] + x_acc_5[end]])
# ax.set_ylim([-1.0, 2.45])
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)
ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500])

# Save the figure
fig.savefig("training_MSE.pdf", bbox_inches="tight")


# Y_ax = -650.0
# fontsz = 13
# linew = 2
# marks = 5
# fontylabel = text("", 12,color=:black).font
# fontzlabel = text("", 12,color=:black).font
# fontzlabel.rotation=90
# plot(x_acc, train_accuracy, ylim=(0,0.0001), color=:black, label=:"SSIM", xlabel="Epoch",left_margin=15Plots.mm, legend=:false, grid=:false, tickfontsize=12, guidefontsize=12, tickfont="Times New Roman")
# plot!(x_acc_5.+x_acc[end], train_accuracy_5, color=:blue, label=:"RMS")
# plot!(x_acc_more.+x_acc_5[end].+x_acc[end], train_accuracy_more, color=:blue, label=:" ")
# plot!(annotations = ([Y_ax], 0.0004,text("RMS", fontylabel)))
# plot!(yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,xtickfontsize=fontsz,ytickfontsize=fontsz)
# savefig("training_MSE.pdf")








# Define font size variable
font_size_axis = 14

# Read and process the data
source = npzread("source_image_neg20_slice_7_5.npy")
generated = npzread("generated_image_neg20_slice_7_5-4.npy")
generated = source[end:-1:1, :]
source = source[end:-1:1, :]

# Define axis scales
crop_Y_plot_fine = range(2245.0, stop=2745.0, length=51)
crop_Z_plot_fine = range(5.0, stop=304.99999999999994, length=31)
Y_scaled = (crop_Y_plot_fine .- 2500) / 126
Z_scaled = (crop_Z_plot_fine .- 90.0) / 90

# Scale the data
generated_scaled = generated .* 8.1

# Plot the generated image
fig, ax = plt.subplots()
cax = ax.imshow(generated_scaled, extent=[Y_scaled[1], Y_scaled[end], Z_scaled[1], Z_scaled[end]], aspect=0.75, cmap="Blues", clim=(0.0, maximum(source) * 8.1))

# Add colorbar with adjusted size
cbar = fig.colorbar(cax, ax=ax, fraction=0.031, pad=0.04)
cbar.set_label("Velocity Deficit (m/s)", fontsize=font_size_axis, fontname="Times New Roman")
cbar.ax.tick_params(labelsize=font_size_axis)  # Set colorbar ticks fontsize

# Set labels and limits
ax.set_xlabel("y/D", fontsize=font_size_axis, fontname="Times New Roman")
# ax.set_ylabel("z*", fontsize=font_size_axis, fontname="Times New Roman")
ax.set_xlim([-2.0, 2])
ax.set_ylim([-1.0, 2.45])

# Set custom ticks
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([-1, 0, 1, 2])
ax.tick_params(axis="both", labelsize=font_size_axis)  # Set axis ticks fontsize

# Hide top and right spines
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)

# Custom annotations
ax.text(-2.5, 0.65, "z*", fontsize=font_size_axis, fontname="Times New Roman")

# Save the figure
# fig.savefig("neg20_slice_7_5_mse_generated.pdf", bbox_inches="tight")
fig.savefig("neg20_slice_7_5_mse_source.pdf", bbox_inches="tight")





# Compute and plot the difference heatmap
diff = abs.(source .- generated)
diff_scaled = diff .* 8.1

fig, ax = plt.subplots()
cax = ax.imshow(diff_scaled, extent=[Y_scaled[1], Y_scaled[end], Z_scaled[1], Z_scaled[end]], aspect=0.75, cmap="Blues", clim=(0.0, maximum(diff) * 8.1))

# Add colorbar with adjusted size
cbar = fig.colorbar(cax, ax=ax, fraction=0.031, pad=0.04)
# cbar.set_label("Difference (m/s)", fontsize=font_size_axis, fontname="Times New Roman")
cbar.ax.tick_params(labelsize=font_size_axis)  # Set colorbar ticks fontsize

# Set labels and limits
ax.set_xlabel("y/D", fontsize=font_size_axis, fontname="Times New Roman")
# ax.set_ylabel("z*", fontsize=font_size_axis, fontname="Times New Roman")
ax.set_xlim([-2.0, 2])
ax.set_ylim([-1.0, 2.45])

# Set custom ticks
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([-1, 0, 1, 2])
ax.tick_params(axis="both", labelsize=font_size_axis)  # Set axis ticks fontsize

# Custom annotations
ax.text(-2.5, 0.65, "z*", fontsize=font_size_axis, fontname="Times New Roman")
ax.text(2.7, 0.65, "\$|\\Delta \\hat U_{\\text{SOWFA}} - \\Delta \\hat U_{\\text{gen}}|\$", fontsize=font_size_axis, fontname="Times New Roman")

# Hide top and right spines
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)

# Save the figure
fig.savefig("neg20_slice_7_5_mse_diff.pdf", bbox_inches="tight")





# crop_Y_plot_fine =  range(2245.0, stop=2745.0, length=51)
# crop_Z_plot_fine = range(5.0, stop=304.99999999999994, length=31)
# source = npzread("source_image_neg20_slice_7_5.npy")
# generated = npzread("generated_image_neg20_slice_7_5-4.npy")
# # crop_Y_plot_fine = 
# data_plot = heatmap((crop_Y_plot_fine.-2500)/126, ((crop_Z_plot_fine.-90.0)/90), generated.*8.1, c=:Blues)
# # data_plot = heatmap((crop_Y_plot_fine.-2500)/126, (crop_Z_plot_fine/90), source.*8.1, c=:Blues)
# plot(data_plot, colorbar_title=" \nVelocity Deficit (m/s)",aspect_ratio=0.7,clim=(0.0, maximum(source)*8.1), xlim=(-2.0, 2), ylim=(-1.0, 2.45), xlabel="y/D")
# plot!(yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,size=((14.5/8)*400,400),bottom_margin=5Plots.mm,left_margin=15Plots.mm, right_margin=15Plots.mm, xtickfontsize=fontsz,ytickfontsize=fontsz)
# plot!(annotations = ([-2.5], 0.75,text("z*", fontylabel)), smooth=false)

# savefig("neg20_slice_7_5_mse_generated.pdf")
# # savefig("neg20_slice_7_5_mse_source.pdf")

# diff = abs.(source.-generated)
# data_plot = heatmap((crop_Y_plot_fine.-2500)/126, ((crop_Z_plot_fine.-90.0)/90), diff.*8.1, c=:Blues)
# # data_plot = heatmap((crop_Y_plot_fine.-2500)/126, (crop_Z_plot_fine/90), source.*8.1, c=:Blues)
# plot(data_plot, colorbar_title=" \n ",aspect_ratio=0.7,clim=(0.0, maximum(diff)*8.1), xlim=(-2.0, 2), ylim=(-1.0, 2.45), xlabel="y/D")
# plot!(yguidefontsize=fontsz,legendfontsize=fontsz,xguidefontsize=fontsz,size=((14.5/8)*400,400),bottom_margin=5Plots.mm,left_margin=15Plots.mm, right_margin=15Plots.mm, xtickfontsize=fontsz,ytickfontsize=fontsz)
# plot!(annotations = ([-2.5], 0.75,text("z*", fontylabel)))
# plot!(annotations = ([3.0], 0.75,text("|\\Delta U_{SOWFA} - \\Delta U_{gen}\\ |", fontzlabel)))
# savefig("neg20_slice_7_5_mse_diff.pdf")






""" Combined Plot of DNN Runs """
source = npzread("source_image_neg20_slice_7_5.npy")
generated = npzread("generated_image_neg20_slice_7_5-4.npy")
generated = source[end:-1:1, :]
source = source[end:-1:1, :]

# Define axis scales
crop_Y_plot_fine = range(2245.0, stop=2745.0, length=51)
crop_Z_plot_fine = range(5.0, stop=304.99999999999994, length=31)
Y_scaled = (crop_Y_plot_fine .- 2500) / 126
Z_scaled = (crop_Z_plot_fine .- 90.0) / 90

# Scale the data
generated_scaled = generated .* 8.1