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

# load data for advanced DNN
train_accuracy_0_1000 = npzread("train_losses_all.npy")
val_acc_0_1000 = npzread("val_losses_all.npy")
train_accuracy_1000_2000 = npzread("train_losses_all_1000_2000.npy")
val_acc_1000_2000 = npzread("val_losses_all_1000_2000.npy")
train_accuracy_2000_3000 = npzread("train_losses_all_2000_3000.npy")
val_acc_2000_3000 = npzread("val_losses_all_2000_3000.npy")

# append the vectors together
append!(train_accuracy_0_1000, train_accuracy_1000_2000, train_accuracy_2000_3000)
append!(val_acc_0_1000, val_acc_1000_2000, val_acc_2000_3000)

training_loss = train_accuracy_0_1000
validation = val_acc_0_1000

# load data for advanced DNN
train_accuracy_0_1000 = npzread("train_losses_mid_0_1000.npy")
val_acc_0_1000 = npzread("val_losses_mid_0_1000.npy")
train_accuracy_1000_2000 = npzread("train_losses_mid_1000_2000.npy")
val_acc_1000_2000 = npzread("val_losses_mid_1000_2000.npy")
train_accuracy_2000_3000 = npzread("train_losses_mid_2000_3000.npy")
val_acc_2000_3000 = npzread("val_losses_mid_2000_3000.npy")

# append the vectors together
append!(train_accuracy_0_1000, train_accuracy_1000_2000, train_accuracy_2000_3000)
append!(val_acc_0_1000, val_acc_1000_2000, val_acc_2000_3000)

training_loss_mid = train_accuracy_0_1000
validation_mid = val_acc_0_1000

# load data for advanced DNN
train_accuracy_0_1000 = npzread("train_losses_large_withbatch_000_1000.npy")
val_acc_0_1000 = npzread("val_losses_large_withbatch_000_1000.npy")
train_accuracy_1000_2000 = npzread("train_losses_large_withbatch_1000_2000.npy")
val_acc_1000_2000 = npzread("val_losses_large_withbatch_1000_2000.npy")
train_accuracy_2000_3000 = npzread("train_losses_large_withbatch_2000_3000.npy")
val_acc_2000_3000 = npzread("val_losses_large_withbatch_2000_3000.npy")

# append the vectors together
append!(train_accuracy_0_1000, train_accuracy_1000_2000, train_accuracy_2000_3000)
append!(val_acc_0_1000, val_acc_1000_2000, val_acc_2000_3000)

training_loss_large = train_accuracy_0_1000
validation_large = val_acc_0_1000

# load data for basic DNN
btrain_accuracy_0_1000 = npzread("train_losses_1000.npy")
bval_acc_0_1000 = npzread("val_losses_1000.npy")
btrain_accuracy_1000_2000 = npzread("train_losses_1000_2000.npy")
bval_acc_1000_2000 = npzread("val_losses_1000_2000.npy")
btrain_accuracy_2000_3000 = npzread("train_losses_2000_3000.npy")
bval_acc_2000_3000 = npzread("val_losses_2000_3000.npy")

append!(btrain_accuracy_0_1000, btrain_accuracy_1000_2000, btrain_accuracy_2000_3000)
append!(bval_acc_0_1000, bval_acc_1000_2000, bval_acc_2000_3000)

training_loss_simple = btrain_accuracy_0_1000
validation_simple = bval_acc_0_1000

# load data for basic DNN
btrain_accuracy_0_1000 = npzread("train_losses_large_new_000_1000_lr_0.001.npy")
bval_acc_0_1000 = npzread("val_losses_large_new_000_1000_lr_0.001.npy")
btrain_accuracy_1000_2000 = npzread("train_losses_large_new_1000_1500_lr_0.001.npy")
bval_acc_1000_2000 = npzread("val_losses_large_new_1000_1500_lr_0.001.npy")
# btrain_accuracy_2000_3000 = npzread("train_losses_2000_3000.npy")
# bval_acc_2000_3000 = npzread("val_losses_2000_3000.npy")

btrain_accuracy_0_1000 = @. (sqrt.(btrain_accuracy_0_1000)/1.8592649698257446)*100.0
bval_acc_0_1000 = @. (sqrt.(bval_acc_0_1000)/1.8765214681625366)*100.0

append!(btrain_accuracy_0_1000, btrain_accuracy_1000_2000)
append!(bval_acc_0_1000, bval_acc_1000_2000)

training_loss_simple = btrain_accuracy_0_1000
validation_simple = bval_acc_0_1000

# convert from MSE to absolute error percentage
# training_loss_simple = @. (sqrt.(training_loss_simple)/1.8672711849212646)*100.0
# validation_simple = @. (sqrt.(validation_simple)/1.8672711849212646)*100.0

# training_loss_simple = @. (sqrt.(training_loss_large)/1.8592649698257446)*100.0
# validation_simple = @. (sqrt.(validation_large)/1.8765214681625366)*100.0

# load data for batch norm, no attention DNN

# Define font size variable
font_size_axis = 14
# Plotting
fig, ax = plt.subplots()

# ax.plot(training_loss, color="dodgerblue", label="Training")
# ax.plot(validation, color="black", label="Validation")
# ax.plot(training_loss_mid, color="dodgerblue", label="Training")
# ax.plot(validation_mid, color="black", label="Validation")
ax.plot(training_loss_simple, color="dodgerblue", label="Training")
ax.plot(validation_simple, color="black", label="Validation")

# Set labels and fonts
ax.set_xlabel("Epoch", fontsize=font_size_axis, fontname="Times New Roman")
ax.set_ylabel("Loss", fontsize=font_size_axis, fontname="Times New Roman")
ax.set_ylabel("Relative Difference (%)", fontsize=font_size_axis, fontname="Times New Roman")
# ax.set_ylim(0, 0.05)
ax.tick_params(axis="both", which="major", labelsize=14, labelcolor="black", width=1.5)

# Custom annotations
# ax.text(-1000.0, 0.00025, "RMS", fontsize=font_size_axis, fontname="Times New Roman")
# ax.text(1550.0, 0.00015, "SSIM", fontsize=font_size_axis, fontname="Times New Roman")
# ax.text(4100.0, 0.0001, "RMS", fontsize=font_size_axis, fontname="Times New Roman", color="dodgerblue")


# Remove grid and legend if necessary
ax.grid(false)
ax.legend().set_visible(true)
ax.set_xlim([0, 1500])
ax.set_ylim([0.0, 20.0])
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)
# ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000])
ax.set_xticks([0, 500, 1000, 1500])

# Save the figure
fig.savefig("Cable_MSE_lr_big_lr_small.pdf", bbox_inches="tight")