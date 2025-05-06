import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_custom_barh(highlights):
    plt.figure(figsize=(10, 5))  # Adjust figure size for better proportions

    # Define bar height dynamically
    bar_height = 0.5  # Slightly larger for better visibility

    # Add highlights
    for highlight in highlights:
        framework_name, highlight_start, highlight_end, highlight_color = highlight
        bar_width = highlight_end - highlight_start
        plt.barh([framework_name], [bar_width], 
                 left=highlight_start, color=highlight_color, edgecolor='black', height=bar_height)
        
        # Add text label at the end of the bar (except for the longest one)
        if highlight_end not in [6.907546, 10.551004]:
            plt.text(highlight_start + bar_width + 0.5, framework_name, f"{highlight_end:.3f}", 
                     va='center', ha='left', fontsize=15, fontweight='bold')

    # Labels and title
    plt.xlabel('Time (s)', fontsize=16, fontweight='bold')
    plt.xlim(0, 35)  # Fixed x-axis size

    plt.title("Latency in a Single Training Round", fontsize=18, fontweight='bold')

    # Add legend for computation and communication times
    computation_patch = mpatches.Patch(color='#DAE8FC', label='Computation Time')  # Blue
    communication_patch = mpatches.Patch(color='#FFF2CC', label='Communication Time')  # Yellow
    plt.legend(handles=[computation_patch, communication_patch], loc='upper right', fontsize=17)

    # Adjust layout to prevent label clipping
    plt.tight_layout()
    plt.xticks(fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')

    # Save and display chart
    plt.savefig('output.pdf', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
            
algo = [
    ('FedAvg\n(large)', 0, 23.231080, '#DAE8FC'),
    ('FedPair', 0, 23.231080, '#DAE8FC'),
    ('HeteroSplit\n(w/o ET)', 0, 10.551004, '#DAE8FC'),
    ('HeteroSplit\n(w/o ET)', 10.551004, 12.631004, '#FFF2CC'), 
    ('HeteroSplit\n(w/ ET)', 0, 6.907546, '#DAE8FC'),
    ('HeteroSplit\n(w/ ET)', 6.907546, 8.987546, '#FFF2CC'),
    ('DepthFL', 0, 6.324312, '#DAE8FC'),
    ('SplitMix', 0, 6.324312, '#DAE8FC'),
    ('FedAvg\n(small)', 0, 6.324312, '#DAE8FC'),
]   

plot_custom_barh(algo)
