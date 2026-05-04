import json
import matplotlib.pyplot as plt
import pandas as pd

def load_and_visualize_mouse_data():
    """Load mouse data and create simple visualizations"""
    
    # Load the data
    data_file = '/Users/nilsness/Desktop/tumii/Research/Projects/Cognition_Agents/ECIS/revisit_study/revisit_study/data_analysis/HAIC_study_all-2.json'
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Extract mouse tracking data
    all_mouse_data = []
    participant_count = 0
    
    for participant in data:
        participant_id = participant['participantId']
        
        # Find mastermind game data
        for answer_key, answer_data in participant['answers'].items():
            if 'mastermind' in answer_key.lower():
                answers = answer_data.get('answer', {})
                
                # Get mouse tracking data
                mouse_data_raw = answers.get('mastermindGame_mouseTrackingData', '[]')
                try:
                    mouse_data = json.loads(mouse_data_raw) if mouse_data_raw else []
                    if mouse_data:  # Only process if there's actual data
                        participant_count += 1
                        print(f"Participant {participant_count}: {len(mouse_data)} mouse movements")
                        
                        # Add participant info to each mouse point
                        for point in mouse_data:
                            point['participant_id'] = participant_id
                            all_mouse_data.append(point)
                        
                        # Create individual participant plot
                        if len(mouse_data) > 10:  # Only plot if sufficient data
                            create_participant_plot(mouse_data, participant_id, participant_count)
                            
                except json.JSONDecodeError:
                    continue
    
    if all_mouse_data:
        # Create combined visualization
        create_combined_plot(all_mouse_data, participant_count)
    else:
        print("No mouse tracking data found in the dataset.")
        print("This might be because:")
        print("- Mouse tracking was added recently")
        print("- The data was collected before this feature was implemented")
        print("- The participants didn't interact with the mouse in a trackable way")

def create_participant_plot(mouse_data, participant_id, participant_num):
    """Create a plot for individual participant mouse movements"""
    
    df = pd.DataFrame(mouse_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Mouse trail
    scatter = ax1.scatter(df['x'], df['y'], c=range(len(df)), cmap='viridis', alpha=0.7, s=20)
    ax1.plot(df['x'], df['y'], alpha=0.3, linewidth=1)
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.set_title(f'Mouse Trail - Participant {participant_num}')
    ax1.invert_yaxis()  # Invert Y-axis to match screen coordinates
    plt.colorbar(scatter, ax=ax1, label='Time sequence')
    
    # Plot 2: Mouse movements by attempt
    if 'attempt' in df.columns:
        attempts = df['attempt'].unique()
        colors = plt.cm.Set3(range(len(attempts)))
        
        for i, attempt in enumerate(sorted(attempts)):
            attempt_data = df[df['attempt'] == attempt]
            ax2.scatter(attempt_data['x'], attempt_data['y'], 
                       c=[colors[i]], label=f'Attempt {attempt}', alpha=0.7, s=15)
        
        ax2.set_xlabel('X Position (pixels)')
        ax2.set_ylabel('Y Position (pixels)')
        ax2.set_title(f'Mouse Movements by Attempt - Participant {participant_num}')
        ax2.invert_yaxis()
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Print some basic stats
    print(f"  - Screen area covered: {df['x'].max() - df['x'].min()} x {df['y'].max() - df['y'].min()} pixels")
    print(f"  - Total movements: {len(df)}")
    if 'attempt' in df.columns:
        print(f"  - Attempts: {df['attempt'].nunique()}")

def create_combined_plot(all_mouse_data, participant_count):
    """Create combined visualization of all mouse data"""
    
    df = pd.DataFrame(all_mouse_data)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: All mouse movements
    participants = df['participant_id'].unique()
    colors = plt.cm.tab10(range(len(participants)))
    
    for i, pid in enumerate(participants):
        participant_data = df[df['participant_id'] == pid]
        ax1.scatter(participant_data['x'], participant_data['y'], 
                   c=[colors[i % len(colors)]], alpha=0.6, s=10, 
                   label=f'P{i+1}')
    
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.set_title(f'All Mouse Movements ({participant_count} participants)')
    ax1.invert_yaxis()
    ax1.legend()
    
    # Plot 2: Movement density heatmap
    ax2.hist2d(df['x'], df['y'], bins=50, cmap='hot')
    ax2.set_xlabel('X Position (pixels)')
    ax2.set_ylabel('Y Position (pixels)')
    ax2.set_title('Movement Density Heatmap')
    ax2.invert_yaxis()
    
    # Plot 3: Movements by attempt (if available)
    if 'attempt' in df.columns:
        attempt_counts = df['attempt'].value_counts().sort_index()
        ax3.bar(attempt_counts.index, attempt_counts.values, alpha=0.7)
        ax3.set_xlabel('Attempt Number')
        ax3.set_ylabel('Number of Mouse Movements')
        ax3.set_title('Mouse Activity by Attempt')
    
    # Plot 4: Target interactions (if available)
    if 'target' in df.columns:
        target_counts = df['target'].value_counts().head(10)
        ax4.barh(range(len(target_counts)), target_counts.values, alpha=0.7)
        ax4.set_yticks(range(len(target_counts)))
        ax4.set_yticklabels(target_counts.index, fontsize=8)
        ax4.set_xlabel('Number of Interactions')
        ax4.set_title('Top 10 Target Elements')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n📊 MOUSE TRACKING SUMMARY:")
    print(f"   • Total participants with mouse data: {participant_count}")
    print(f"   • Total mouse movements recorded: {len(df)}")
    print(f"   • Average movements per participant: {len(df)/participant_count:.1f}")
    
    if 'attempt' in df.columns:
        print(f"   • Total attempts across all participants: {df['attempt'].nunique()}")
    
    # Screen coverage
    x_range = df['x'].max() - df['x'].min()
    y_range = df['y'].max() - df['y'].min()
    print(f"   • Screen area coverage: {x_range:.0f} x {y_range:.0f} pixels")

if __name__ == "__main__":
    load_and_visualize_mouse_data()
