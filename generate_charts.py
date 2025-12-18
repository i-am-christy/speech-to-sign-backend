import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# ASR Training Loss (Figure 4.4) 
def plot_training_loss():
    # Data from your ASR Evaluation Results
    steps = [100, 200, 300, 400, 500]
    train_loss = [0.3308, 0.0346, 0.0067, 0.0016, 0.0009]
    val_loss = [0.2995, 0.1893, 0.1947, 0.2137, 0.2203]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    plt.plot(steps, val_loss, 'r--s', label='Validation Loss', linewidth=2, markersize=6)
    
    plt.title('ASR Model Fine-Tuning Loss Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Training Steps')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Figure_4.4_ASR_Loss.png', dpi=300)
    print("Generated Figure 4.4")

# User Study Results (Figure 4.11) 
def plot_user_study():
    categories = ['Comprehensibility', 'Naturalness', 'Usefulness']
    scores = [3.9, 3.8, 4.2]
    #errors = [0.74/2, 0.84/2, 0.45/2] # Standard Error bars

    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, scores,  capsize=10, 
                   color=['#4c72b0', '#55a868', '#c44e52'], alpha=0.9, width=0.6)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height}/5.0', ha='center', va='bottom', fontweight='bold')

    plt.title('User Evaluation: Mean Opinion Scores (MOS)', fontsize=14, fontweight='bold')
    plt.ylabel('Score (1-5 Scale)')
    plt.ylim(0, 5.5)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Figure_4.11_User_Study.png', dpi=300)
    print("Generated Figure 4.11")

if __name__ == "__main__":
    plot_training_loss()
    plot_user_study()