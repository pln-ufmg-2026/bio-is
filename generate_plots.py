import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality style configurations
sns.set_theme(style='whitegrid', context='talk')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#CBD5E0',
    'grid.color': '#E2E8F0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.facecolor': 'white',
    'legend.edgecolor': '#E2E8F0',
    'figure.titlesize': 18,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

# Color palette definition for the 5 configurations
CONFIG_COLORS = {
    'FT (Baseline)': '#2D3748',      # Dark Slate Grey
    'IS (biois)': '#0D9488',         # Teal
    'IS (drop3)': '#E11D48',         # Pink-Red
    'CL+IS (biois)': '#2563EB',      # Royal Blue
    'CL+IS (drop3)': '#7C3AED'       # Purple
}

# Iterations to analyze
ITERATIONS = [
    #'iter0', 'iter1', 
    'iter2', 
    'iter3', 'iter4']

def parse_filename(filename):
    """
    Parses a filename to extract the dataset name and the instance selection algorithm.
    """
    name = filename.lower()
    
    # Dataset mapping
    if "aisopos" in name:
        dataset = "Aisopos"
    elif "mr" in name:
        dataset = "MR"
    elif "subj" in name:
        dataset = "Subj"
    elif "vader" in name:
        dataset = "Vader Movie"
    else:
        # Fallback to name minus the _out.csv suffix
        dataset = filename.split("_out.csv")[0]

    # Algorithm mapping
    if "biois" in name:
        algo = "biois"
    elif "drop3" in name:
        algo = "drop3"
    else:
        algo = None
        
    return dataset, algo

def get_config_name(setup, algo):
    """
    Standardizes configuration names.
    """
    if setup == 'FT':
        return 'FT (Baseline)'
    elif algo:
        return f'{setup} ({algo})'
    else:
        return setup

def load_results(results_dir='results'):
    """
    Crawls the results directory and aggregates summary and epoch-level data.
    """
    summary_records = []
    epoch_records = []
    
    setups = ['FT', 'IS', 'CL+IS']
    
    for iter_name in ITERATIONS:
        iter_path = os.path.join(results_dir, iter_name)
        if not os.path.isdir(iter_path):
            print(f"Directory {iter_path} not found. Skipping...")
            continue
            
        for setup in setups:
            setup_path = os.path.join(iter_path, setup)
            if not os.path.isdir(setup_path):
                continue
                
            # Find all output files
            csv_files = glob.glob(os.path.join(setup_path, '*_out.csv'))
            for csv_file in csv_files:
                filename = os.path.basename(csv_file)
                # Skip batch files
                if '_batch_' in filename:
                    continue
                    
                dataset, algo = parse_filename(filename)
                config = get_config_name(setup, algo)
                
                try:
                    df = pd.read_csv(csv_file)
                    if df.empty:
                        print(f"File {csv_file} is empty. Skipping...")
                        continue
                        
                    # Check for required columns
                    required = ['epoch', 'eval_f1', 'eval_accuracy', 'accumulated_time_seconds']
                    if not all(col in df.columns for col in required):
                        print(f"File {csv_file} is missing required columns. Skipping...")
                        continue
                        
                    # Extract epoch-level details
                    if 'fold' in df.columns:
                        # Handle multiple folds if present by grouping by epoch and averaging metrics
                        df_sorted = df.sort_values(['fold', 'epoch'])
                        epoch_df = df_sorted.groupby('epoch').agg({
                            'eval_f1': 'mean',
                            'eval_accuracy': 'mean',
                            'eval_loss': 'mean' if 'eval_loss' in df.columns else 'first',
                            'accumulated_time_seconds': 'mean'
                        }).reset_index()
                        
                        # Find best epoch metrics per fold and average them
                        fold_metrics = []
                        for fold, fold_df in df.groupby('fold'):
                            fold_df = fold_df.sort_values('epoch')
                            if not fold_df.empty:
                                best_idx = fold_df['eval_f1'].idxmax()
                                best_row = fold_df.loc[best_idx]
                                total_time = fold_df['accumulated_time_seconds'].iloc[-1]
                                fold_metrics.append({
                                    'f1': best_row['eval_f1'],
                                    'accuracy': best_row['eval_accuracy'],
                                    'total_time': total_time
                                })
                        if fold_metrics:
                            best_f1 = np.mean([m['f1'] for m in fold_metrics])
                            best_acc = np.mean([m['accuracy'] for m in fold_metrics])
                            total_time = np.mean([m['total_time'] for m in fold_metrics])
                        else:
                            best_f1 = epoch_df['eval_f1'].max()
                            best_acc = epoch_df.loc[epoch_df['eval_f1'].idxmax(), 'eval_accuracy']
                            total_time = epoch_df['accumulated_time_seconds'].iloc[-1]
                    else:
                        df_sorted = df.sort_values('epoch')
                        best_idx = df_sorted['eval_f1'].idxmax()
                        best_row = df_sorted.loc[best_idx]
                        best_f1 = best_row['eval_f1']
                        best_acc = best_row['eval_accuracy']
                        total_time = df_sorted['accumulated_time_seconds'].iloc[-1]
                        epoch_df = df_sorted
                        
                    summary_records.append({
                        'iteration': iter_name,
                        'setup': setup,
                        'algorithm': algo if algo else 'None',
                        'config': config,
                        'dataset': dataset,
                        'best_f1': best_f1,
                        'best_accuracy': best_acc,
                        'total_time': total_time
                    })
                    
                    for _, row in epoch_df.iterrows():
                        epoch_records.append({
                            'iteration': iter_name,
                            'setup': setup,
                            'algorithm': algo if algo else 'None',
                            'config': config,
                            'dataset': dataset,
                            'epoch': int(row['epoch']),
                            'eval_f1': row['eval_f1'],
                            'eval_accuracy': row['eval_accuracy'],
                            'eval_loss': row['eval_loss'] if 'eval_loss' in row and not pd.isna(row['eval_loss']) else 0.0,
                            'accumulated_time_seconds': row['accumulated_time_seconds']
                        })
                except Exception as e:
                    print(f"Error parsing file {csv_file}: {e}")
                    
    return pd.DataFrame(summary_records), pd.DataFrame(epoch_records)

def plot_iteration_bar_charts(df_summary, iter_name, output_dir):
    """
    Generates bar charts for F1, Accuracy, and Training Time for a single iteration.
    """
    df_iter = df_summary[df_summary['iteration'] == iter_name]
    if df_iter.empty:
        return
        
    datasets = sorted(df_iter['dataset'].unique())
    configs = [c for c in CONFIG_COLORS.keys() if c in df_iter['config'].unique()]
    
    # 1. Performance (Eval F1) Bar Chart
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_iter,
        x='dataset',
        y='best_f1',
        hue='config',
        hue_order=configs,
        palette=CONFIG_COLORS,
        edgecolor='black',
        linewidth=0.8
    )
    plt.title(f'Best Eval F1 Score by Configuration - {iter_name}', pad=20)
    plt.xlabel('Dataset', labelpad=10)
    plt.ylabel('Eval F1 Score', labelpad=10)
    plt.ylim(0, 1.05)
    plt.legend(title='Configuration', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, xytext=(0, 3),
                        textcoords='offset points')
                        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_f1.png'), dpi=300)
    plt.close()
    
    # 2. Performance (Eval Accuracy) Bar Chart
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_iter,
        x='dataset',
        y='best_accuracy',
        hue='config',
        hue_order=configs,
        palette=CONFIG_COLORS,
        edgecolor='black',
        linewidth=0.8
    )
    plt.title(f'Best Eval Accuracy by Configuration - {iter_name}', pad=20)
    plt.xlabel('Dataset', labelpad=10)
    plt.ylabel('Eval Accuracy', labelpad=10)
    plt.ylim(0, 1.05)
    plt.legend(title='Configuration', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, xytext=(0, 3),
                        textcoords='offset points')
                        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_accuracy.png'), dpi=300)
    plt.close()
    
    # 3. Training Time (Minutes) Bar Chart
    df_iter_time = df_iter.copy()
    df_iter_time['total_time_minutes'] = df_iter_time['total_time'] / 60.0
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_iter_time,
        x='dataset',
        y='total_time_minutes',
        hue='config',
        hue_order=configs,
        palette=CONFIG_COLORS,
        edgecolor='black',
        linewidth=0.8
    )
    plt.title(f'Total Training Time by Configuration - {iter_name}', pad=20)
    plt.xlabel('Dataset', labelpad=10)
    plt.ylabel('Training Time (Minutes)', labelpad=10)
    plt.legend(title='Configuration', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Annotate time on bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}m',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, xytext=(0, 3),
                        textcoords='offset points')
                        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time.png'), dpi=300)
    plt.close()

def plot_iteration_learning_curves(df_epochs, iter_name, output_dir):
    """
    Generates 2x2 grid line plots for F1 and Loss progression.
    """
    df_iter = df_epochs[df_epochs['iteration'] == iter_name]
    if df_iter.empty:
        return
        
    datasets = sorted(df_iter['dataset'].unique())
    configs = [c for c in CONFIG_COLORS.keys() if c in df_iter['config'].unique()]
    
    # 1. Eval F1 curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        df_ds = df_iter[df_iter['dataset'] == dataset]
        
        for config in configs:
            df_cfg = df_ds[df_ds['config'] == config].sort_values('epoch')
            if not df_cfg.empty:
                ax.plot(
                    df_cfg['epoch'],
                    df_cfg['eval_f1'],
                    marker='o',
                    markersize=4,
                    label=config,
                    color=CONFIG_COLORS[config],
                    linewidth=1.5
                )
        ax.set_title(dataset)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Eval F1')
        ax.set_ylim(0, 1.05)
        
    # Place legend in a clean way or in the empty space if any
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(configs), bbox_to_anchor=(0.5, -0.02))
    
    fig.suptitle(f'Eval F1 Convergence Curves - {iter_name}', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'learning_curves_f1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Eval Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        df_ds = df_iter[df_iter['dataset'] == dataset]
        
        for config in configs:
            df_cfg = df_ds[df_ds['config'] == config].sort_values('epoch')
            if not df_cfg.empty:
                ax.plot(
                    df_cfg['epoch'],
                    df_cfg['eval_loss'],
                    marker='s',
                    markersize=4,
                    label=config,
                    color=CONFIG_COLORS[config],
                    linewidth=1.5
                )
        ax.set_title(dataset)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Eval Loss')
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(configs), bbox_to_anchor=(0.5, -0.02))
    
    fig.suptitle(f'Eval Loss Convergence Curves - {iter_name}', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'learning_curves_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Bar plot version for Eval F1 Convergence
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=False)
    axes = axes.flatten()
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        df_ds = df_iter[df_iter['dataset'] == dataset]
        sns.barplot(
            data=df_ds,
            x='epoch',
            y='eval_f1',
            hue='config',
            hue_order=configs,
            palette=CONFIG_COLORS,
            ax=ax,
            edgecolor='black',
            linewidth=0.5
        )
        ax.set_title(dataset)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Eval F1')
        ax.set_ylim(0, 1.15)
        ax.legend().remove()
        
        # Annotate values
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom',
                            fontsize=7, xytext=(0, 2),
                            textcoords='offset points',
                            rotation=90)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(configs), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'Eval F1 Convergence (Bar Chart) - {iter_name}', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'learning_curves_f1_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Bar plot version for Eval Loss Convergence
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=False)
    axes = axes.flatten()
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        df_ds = df_iter[df_iter['dataset'] == dataset]
        sns.barplot(
            data=df_ds,
            x='epoch',
            y='eval_loss',
            hue='config',
            hue_order=configs,
            palette=CONFIG_COLORS,
            ax=ax,
            edgecolor='black',
            linewidth=0.5
        )
        ax.set_title(dataset)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Eval Loss')
        ax.set_ylim(0, df_ds['eval_loss'].max() * 1.15 if not df_ds.empty and df_ds['eval_loss'].max() > 0 else 1.15)
        ax.legend().remove()
        
        # Annotate values
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom',
                            fontsize=7, xytext=(0, 2),
                            textcoords='offset points',
                            rotation=90)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(configs), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'Eval Loss Convergence (Bar Chart) - {iter_name}', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'learning_curves_loss_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_overall_comparisons(df_summary, output_dir):
    """
    Generates comparison plots across iterations (iter0, iter1, iter2) for each configuration.
    """
    # Group by iteration and configuration to compute average across datasets
    df_avg = df_summary.groupby(['iteration', 'config']).agg({
        'best_f1': 'mean',
        'best_accuracy': 'mean',
        'total_time': 'mean'
    }).reset_index()
    
    # Standardize iteration ordering
    iter_order = {iter_name: idx for idx, iter_name in enumerate(ITERATIONS)}
    df_avg['iter_idx'] = df_avg['iteration'].map(iter_order)
    df_avg = df_avg.sort_values('iter_idx')
    
    configs = [c for c in CONFIG_COLORS.keys() if c in df_avg['config'].unique()]
    
    # 1. Progression of F1
    plt.figure(figsize=(10, 6))
    for config in configs:
        df_cfg = df_avg[df_avg['config'] == config]
        if not df_cfg.empty:
            plt.plot(
                df_cfg['iteration'],
                df_cfg['best_f1'],
                marker='o',
                markersize=8,
                label=config,
                color=CONFIG_COLORS[config],
                linewidth=2.5
            )
    plt.title('Average F1 Score Progression across Iterations', pad=20)
    plt.xlabel('Experiment Iteration', labelpad=10)
    plt.ylabel('Average Eval F1 (All Datasets)', labelpad=10)
    plt.ylim(0, 1.05)
    plt.legend(title='Configuration', loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_f1_progression.png'), dpi=300)
    plt.close()
    
    # 2. Progression of Accuracy
    plt.figure(figsize=(10, 6))
    for config in configs:
        df_cfg = df_avg[df_avg['config'] == config]
        if not df_cfg.empty:
            plt.plot(
                df_cfg['iteration'],
                df_cfg['best_accuracy'],
                marker='s',
                markersize=8,
                label=config,
                color=CONFIG_COLORS[config],
                linewidth=2.5
            )
    plt.title('Average Accuracy Progression across Iterations', pad=20)
    plt.xlabel('Experiment Iteration', labelpad=10)
    plt.ylabel('Average Eval Accuracy (All Datasets)', labelpad=10)
    plt.ylim(0, 1.05)
    plt.legend(title='Configuration', loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_accuracy_progression.png'), dpi=300)
    plt.close()
    
    # 3. Progression of Training Time (Minutes)
    plt.figure(figsize=(10, 6))
    for config in configs:
        df_cfg = df_avg[df_avg['config'] == config]
        if not df_cfg.empty:
            plt.plot(
                df_cfg['iteration'],
                df_cfg['total_time'] / 60.0,
                marker='^',
                markersize=8,
                label=config,
                color=CONFIG_COLORS[config],
                linewidth=2.5
            )
    plt.title('Average Training Time Progression across Iterations', pad=20)
    plt.xlabel('Experiment Iteration', labelpad=10)
    plt.ylabel('Average Training Time (Minutes)', labelpad=10)
    plt.legend(title='Configuration', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_time_progression.png'), dpi=300)
    plt.close()

    # 4. Progression of F1 (Bar Plot)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_avg,
        x='iteration',
        y='best_f1',
        hue='config',
        hue_order=configs,
        palette=CONFIG_COLORS,
        edgecolor='black',
        linewidth=0.8
    )
    plt.title('Average F1 Score Progression (Bar Chart)', pad=20)
    plt.xlabel('Experiment Iteration', labelpad=10)
    plt.ylabel('Average Eval F1 (All Datasets)', labelpad=10)
    plt.ylim(0, 1.15)
    plt.legend(title='Configuration', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Annotate values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, xytext=(0, 3),
                        textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_f1_progression_bar.png'), dpi=300)
    plt.close()

    # 5. Progression of Accuracy (Bar Plot)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_avg,
        x='iteration',
        y='best_accuracy',
        hue='config',
        hue_order=configs,
        palette=CONFIG_COLORS,
        edgecolor='black',
        linewidth=0.8
    )
    plt.title('Average Accuracy Progression (Bar Chart)', pad=20)
    plt.xlabel('Experiment Iteration', labelpad=10)
    plt.ylabel('Average Eval Accuracy (All Datasets)', labelpad=10)
    plt.ylim(0, 1.15)
    plt.legend(title='Configuration', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, xytext=(0, 3),
                        textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_accuracy_progression_bar.png'), dpi=300)
    plt.close()

    # 6. Progression of Training Time (Bar Plot)
    df_avg_time = df_avg.copy()
    df_avg_time['total_time_minutes'] = df_avg_time['total_time'] / 60.0
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_avg_time,
        x='iteration',
        y='total_time_minutes',
        hue='config',
        hue_order=configs,
        palette=CONFIG_COLORS,
        edgecolor='black',
        linewidth=0.8
    )
    plt.title('Average Training Time Progression (Bar Chart)', pad=20)
    plt.xlabel('Experiment Iteration', labelpad=10)
    plt.ylabel('Average Training Time (Minutes)', labelpad=10)
    plt.legend(title='Configuration', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}m',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, xytext=(0, 3),
                        textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_time_progression_bar.png'), dpi=300)
    plt.close()

def main():
    print("Loading results and crawling CSVs...")
    df_summary, df_epochs = load_results()
    
    if df_summary.empty:
        print("No results parsed. Please check that 'results/' contains iter0/iter1/iter2 subfolders with correct CSV structure.")
        return
        
    print(f"Aggregated summary records for {len(df_summary)} runs.")
    print(f"Aggregated epoch records for {len(df_epochs)} rows.")
    
    # Generate subplots per iteration
    for iter_name in ITERATIONS:
        print(f"Generating plots for {iter_name}...")
        iter_dir = os.path.join('plots', iter_name)
        os.makedirs(iter_dir, exist_ok=True)
        
        plot_iteration_bar_charts(df_summary, iter_name, iter_dir)
        plot_iteration_learning_curves(df_epochs, iter_name, iter_dir)
        
    print("Generating overall comparison plots...")
    os.makedirs('plots/overall', exist_ok=True)
    plot_overall_comparisons(df_summary, 'plots/overall')
    
    latest_iter = ITERATIONS[-1]
    print(f"\n--- Summary Performance Table ({latest_iter}) ---")
    df_latest = df_summary[df_summary['iteration'] == latest_iter]
    if not df_latest.empty:
        pivot_f1 = df_latest.pivot(index='config', columns='dataset', values='best_f1')
        print("\nEval F1 Scores:")
        print(pivot_f1.to_string())
        
        pivot_time = df_latest.pivot(index='config', columns='dataset', values='total_time')
        print("\nTraining Time (Seconds):")
        print(pivot_time.to_string())
        
    print("\nAll plots generated successfully under 'plots/' directory!")

if __name__ == '__main__':
    main()
