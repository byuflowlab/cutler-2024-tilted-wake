# I had to run the following

# using Conda
# Conda.add("onnxruntime")
# Conda.add("onnx")

using PyCall
using JSON
using LinearAlgebra

function DNN_cabling(opt_dist)
    # Import the necessary Python modules
    onnxruntime = pyimport("onnxruntime")

    # Load the ONNX model using ONNX Runtime in Python
    session = onnxruntime.InferenceSession("windfarm_net.onnx")

    # Convert opt_dist to a Float32 matrix
    opt_dist_matrix = convert(Matrix{Float32}, reshape(opt_dist, 51, 51))

    # Extract opt_dist_turb (upper triangular without diagonal)
    opt_dist_turb = opt_dist_matrix[1:end-1, 1:end-1] ./ 10000  # Remove the last row and last column
    upper_triangular_indices = triu(ones(Bool, size(opt_dist_turb)), 0)  # Upper triangular with diagonal
    opt_dist_turb = opt_dist_turb[upper_triangular_indices]  # Extract upper triangular part

    # Extract opt_dist_sub (distances between turbines and substation)
    opt_dist_sub = opt_dist_matrix[1:end-1, 1] ./ 10000  # First column after removing the substation

    # Reshape both to ensure they are 2D
    opt_dist_turb = reshape(opt_dist_turb, 1, length(opt_dist_turb))  # Reshape to (1, features)
    opt_dist_sub = reshape(opt_dist_sub, 1, length(opt_dist_sub))    # Reshape to (1, features)

    # Ensure the data type is Float32 for both
    opt_dist_turb = Float32.(opt_dist_turb)
    opt_dist_sub = Float32.(opt_dist_sub)

    # Prepare input data for the ONNX model
    input_data = Dict("opt_dist_turb" => opt_dist_turb, "opt_dist_sub" => opt_dist_sub)

    # Perform inference
    result = session.run(["total_cable_cost"], input_data)

    # Extract and print the output
    total_cable_cost = result[1]
    # print("total_cable_cost: ", total_cable_cost*10000000, "\n")

    # adjust cost
    total_cost = total_cable_cost[1]*10000000
    return total_cost
end

# Load JSON data
data = JSON.parsefile("Local_data_50turb_147.json.json")  # Adjust the path as needed

opt_dist = data[1]["opt_dist"]
opt_val = data[1]["opt_val"]

total_cost_est = DNN_cabling(opt_dist)
print("total_cost_est: ", total_cost_est, "\n")
print("opt_val: ", opt_val/1000, "\n")

relative_diff = ((abs(opt_val/1000-total_cost_est[1]))/(opt_val/1000))*100.0
print("relative_diff: ", relative_diff, "\n")


# data = JSON.parsefile("Local_data_50turb_235.json.json")  # Adjust the path as needed
# data = JSON.parsefile("Local_data_50turb_123.json.json")  # Adjust the path as needed
# data = JSON.parsefile("Local_data_50turb_63.json.json")  # Adjust the path as needed
# data = JSON.parsefile("Local_data_50turb_206.json.json")  # Adjust the path as needed

# # Extract opt_dist_turb (upper triangular without diagonal)
# opt_dist_turb = opt_dist_matrix[2:end, 2:end] ./ 10000  # Remove the last row and last column
# upper_triangular_indices = triu(ones(Bool, size(opt_dist_turb)), 0)  # Upper triangular with diagonal
# opt_dist_turb = opt_dist_turb[upper_triangular_indices]  # Extract upper triangular part

# # Extract opt_dist_sub (distances between turbines and substation)
# opt_dist_sub = opt_dist_matrix[2:end, 1] ./ 10000  # First column after removing the substation