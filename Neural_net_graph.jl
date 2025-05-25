# using Graphs  # Use Graphs.jl for graph representation
# using GraphPlot  # For plotting the graph
# using Colors  # For defining colors like black and lightblue
# using Compose
# # using Cairo  # Import Cairo for PDF functionality
# using Plots

# # Define a simple graph structure for the neural network
# function create_windfarm_net_graph()
#     g = DiGraph()

#     # Add 18 vertices (input layers, turbine layers, substation layers, combined layers, output)
#     add_vertices!(g, 18)

#     # Define the connections (edges)
#     # Turbine layers
#     add_edge!(g, 1, 3)  # 1275 -> 1024
#     add_edge!(g, 3, 4)  # 1024 -> 512
#     add_edge!(g, 4, 5)  # 512 -> 256
#     add_edge!(g, 5, 6)  # 256 -> 128

#     # Substation layers
#     add_edge!(g, 2, 8)  # 50 -> 256
#     add_edge!(g, 8, 9)  # 256 -> 128
#     add_edge!(g, 9, 10) # 128 -> 64

#     # Combined layers
#     add_edge!(g, 6, 11)  # Turbine final layer -> Combined
#     add_edge!(g, 10, 11) # Substation final layer -> Combined
#     add_edge!(g, 11, 12) # 192 -> 512
#     add_edge!(g, 12, 13) # 512 -> 256
#     add_edge!(g, 13, 14) # 256 -> 128
#     add_edge!(g, 14, 15) # 128 -> 64
#     add_edge!(g, 15, 16) # 64 -> 32
#     add_edge!(g, 16, 17) # 32 -> 1

#     # Output
#     add_edge!(g, 17, 18)

#     return g
# end

# # Create the graph for the neural network
# g = create_windfarm_net_graph()

# # Labels for the layers (match the 18 vertices)
# labels = ["opt_dist_turb (1275)", "opt_dist_sub (50)",
#           "Turbine: Linear (1024)", "Turbine: Linear (512)", "Turbine: Linear (256)", "Turbine: Linear (128)",
#           "Substation: Linear (256)", "Substation: Linear (128)", "Substation: Linear (64)",
#           "Combined: Linear (512)", "Combined: Linear (256)", "Combined: Linear (128)",
#           "Combined: Linear (64)", "Combined: Linear (32)", "Combined: Linear (1)",
#           "Final: Output Layer", "", ""]

# # Generate layout coordinates
# layout = spring_layout(g)

# gplot(g, layout[1], layout[2], 
#       nodelabel=labels, 
#       nodefillc=RGB(0.68, 0.85, 0.9),  # light blue color
#       edgestrokec=RGB(0, 0, 0),  # black color
#       nodelabelsize=10)

# savefig("neural_network_graph.pdf")

# # # Plot the neural network graph with the computed layout
# # p = gplot(g, layout[1], layout[2], 
# #       nodelabel=labels, 
# #       nodefillc=RGB(0.68, 0.85, 0.9),  # light blue color
# #       edgestrokec=RGB(0, 0, 0),  # black color
# #       nodelabelsize=10)

# # p.save("neural_net_graph.pdf")

using Graphs
using Plots

# Your existing graph setup
g = SimpleDiGraph(10)  # Example graph
# Add edges and nodes as needed...

# Define the layout (You can use any layout you prefer)
layout = spring_layout(g)
labels = ["Node 1", "Node 2", "Node 3", "Node 4", "Node 5", "Node 6", "Node 7", "Node 8", "Node 9", "Node 10"]

# Create a scatter plot for the nodes
scatter = scatter(layout[:, 1], layout[:, 2], label="", color=:lightblue, size=8)

# Plot edges
for edge in edges(g)
    src = src(edge)
    dst = dst(edge)
    plot!([layout[src, 1], layout[dst, 1]], [layout[src, 2], layout[dst, 2]], color=:black)
end

# Add labels to the nodes
for i in 1:length(labels)
    annotate!(scatter, layout[i, 1], layout[i, 2], text(labels[i], 10))
end

# Save the plot as a PDF
savefig("neural_network_graph.pdf")