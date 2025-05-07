import matplotlib.pyplot as plt

def plot_custom_barh(highlights):

    plt.figure(figsize=(8, 2.5))  # 簡報用

    # Add highlights
    for highlight in highlights:
        framework_name, highlight_start, highlight_end, highlight_color = highlight
        plt.barh([framework_name], [highlight_end - highlight_start], 
                 left=highlight_start, color=highlight_color, edgecolor='black', height=0.5)

    # Add labels and title
    plt.xlabel('Unit time', fontsize=12, fontweight='bold', fontname='Times New Roman')
    #plt.title('Latency')

    # Set fixed x-axis range
    plt.xlim(0, 8.3)  # Fixed x-axis size from 0 to 8 簡報用

    #plt.title("Propose", fontsize=18, fontweight='bold', fontname='Times New Roman')
    plt.title("", fontsize=18, fontweight='bold', fontname='Times New Roman')

    plt.text(4.5, 0.5, '2', fontsize=18, fontweight='bold', color='black', ha='center', va='center', fontname='Times New Roman')

    # Adjust layout to prevent label clipping
    # plt.tight_layout()       # Automatically adjust layout
    plt.subplots_adjust(bottom=0.3)  # Ensure x-axis label is visible
    plt.xticks(fontsize=15, fontweight='bold', fontname='Times New Roman')
    plt.yticks(fontsize=15, fontweight='bold', fontname='Times New Roman')

    # Show the chart
    plt.savefig('output.png', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()

# FL
FL = [
    ('Weak', 0, 8, '#D6E8D5'),
    ('Weak', 4, 8, '#D6E8D5'),
    ('Strong', 0, 2, '#DAE8FC'),  # Highlight a segment of Method A
    ('Strong', 1, 2, '#DAE8FC'),
]

# DepthFL
DepthFL = [
    ('Weak', 0, 1, '#D6E8D5'),
    ('Weak', 1, 2, '#D6E8D5'),
    ('Strong', 0, 1, '#DAE8FC'),  # Highlight a segment of Method A
    ('Strong', 1, 2, '#DAE8FC'),
]

# Propose
Propose = [
    ('Weak', 0, 1, '#D6E8D5'),
    ('Weak', 2.5, 3.5, '#D6E8D5'),
    ('Strong', 0, 0.25, '#DAE8FC'),  # Highlight a segment of Method A
    ('Strong', 0.25, 0.5, '#DAE8FC'),
    ('Strong', 1, 1.75, '#D6E8D5'),
    ('Strong', 1.75, 2.5, '#D6E8D5'),
]

# Propose_imagination
Propose_imagination = [
    ('Weak', 0, 1, '#D6E8D5'),
    ('Weak', 1, 2, '#D6E8D5'),
    ('Strong', 0, 0.25, '#DAE8FC'),  # Highlight a segment of Method A
    ('Strong', 0.25, 0.5, '#DAE8FC'),
    ('Strong', 0.5, 1.25, '#D6E8D5'),
    ('Strong', 1.25, 2, '#D6E8D5'),
]

Propose_2_1 = [
    ('Weak', 0, 1, '#D6E8D5'),
    ('Weak', 2, 3, '#D6E8D5'),
    ('Middle', 0, 0.5, '#FFF2CC'),  # Highlight a segment of Method A
    ('Middle', 0.5, 1, '#FFF2CC'),
    ('Middle', 1, 1.5, '#D6E8D5'),
    ('Middle', 1.5, 2, '#D6E8D5'),
]

# Propose with Early Transmission
ProposeET = [
    ('Weak', 0, 0.25, '#D6E8D5'),
    ('Weak', 2.125, 2.375, '#D6E8D5'),
    ('Strong', 0, 0.25, '#DAE8FC'),  # Highlight a segment of Method A
    ('Strong', 0.25, 1.1875, '#D6E8D5'),
    ('Strong', 1.1875, 2.125, '#D6E8D5'),
    ('Strong', 2.125, 2.375, '#DAE8FC'),
]

Propose_2_1ET = [
    ('Weak', 0, 0.5, '#D6E8D5'),
    ('Weak', 2, 2.5, '#D6E8D5'),
    ('Middle', 0, 0.5, '#FFF2CC'),  # Highlight a segment of Method A
    ('Middle', 0.5, 1.25, '#D6E8D5'),
    ('Middle', 1.25, 2, '#D6E8D5'),
    ('Middle', 2, 2.5, '#FFF2CC'),
]

# Weak #D6E8D5  Middle #FFF2CC  Strong #DAE8FC
A_B_C = [
    ('Weak', 0, 0.25, '#D6E8D5'),
    ('Weak', 2, 2.25, '#D6E8D5'),
    ('Middle', 0, 0.25, '#FFF2CC'),  # Highlight a segment of Method A
    ('Middle', 0.25, 1.125, '#D6E8D5'),
    ('Middle', 1.125, 2, '#D6E8D5'),
    ('Middle', 2, 2.25, '#FFF2CC'),
    ('Strong', 0, 0.25, '#DAE8FC'),  # Highlight a segment of Method A
    ('Strong', 0.25, 1, '#FFF2CC'),
    ('Strong', 1, 1.75, '#FFF2CC'),
    ('Strong', 1.75, 2, '#DAE8FC'),
]

Propose_2_4 = [
    ('Strong', 0, 0.25, '#DAE8FC'),
    ('Strong', 1.5, 1.75, '#DAE8FC'),
    ('Middle', 0, 0.5, '#FFF2CC'),  # Highlight a segment of Method A
    ('Middle', 0.5, 1, '#DAE8FC'),
    ('Middle', 1, 1.5, '#DAE8FC'),
    ('Middle', 1.5, 2, '#FFF2CC'),

]


plot_custom_barh(Propose_2_4)
