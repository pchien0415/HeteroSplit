import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_custom_barh(highlights):
    plt.figure(figsize=(8, 4))  # Adjust figure size for better proportions

    # Define bar height dynamically
    bar_height = 0.3  # Slightly larger for better visibility

    # Add highlights
    for highlight in highlights:
        framework_name, highlight_start, highlight_end, highlight_color = highlight
        bar_width = highlight_end - highlight_start
        plt.barh([framework_name], [bar_width], 
                 left=highlight_start, color=highlight_color, edgecolor='black', height=bar_height)
        
        # Add text label at the end of the bar (except for the longest one)
        if highlight_end != 6.907546:
            plt.text(highlight_start + bar_width + 0.1, framework_name, f"{highlight_end:.3f}", 
                     va='center', ha='left', fontsize=12, fontweight='bold')

    # Labels and title
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.xlim(0, 4)  # Fixed x-axis size

    plt.title("Impact of Transmission Rate", fontsize=18, fontweight='bold')


    # Adjust layout to prevent label clipping
    plt.tight_layout()
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11, fontweight='bold')

    # Save and display chart
    plt.savefig('output.png', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
            
algo = [
    ('Wifi 6', 0, 0.520, '#DAE8FC'),
    ('Wifi 5', 0, 0.694, '#DAE8FC'),
    ('6G', 0, 1.250, '#DAE8FC'),
    ('Wifi 4', 0, 2.083, '#DAE8FC'),
]   

plot_custom_barh(algo)
