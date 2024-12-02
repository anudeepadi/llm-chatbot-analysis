import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class ResponseVisualizer:
    def __init__(self):
        self.style = 'seaborn'
        plt.style.use(self.style)
        self.default_figsize = (12, 6)

    def plot_length_distribution(self, lengths, title='Response Length Distribution'):
        """Plot distribution of response lengths."""
        plt.figure(figsize=self.default_figsize)
        sns.histplot(lengths, kde=True)
        plt.title(title)
        plt.xlabel('Length')
        plt.ylabel('Count')
        return plt.gcf()

    def plot_quality_metrics(self, metrics_df, title='Response Quality Metrics'):
        """Plot various quality metrics."""
        plt.figure(figsize=self.default_figsize)
        
        # Create box plots for each metric
        sns.boxplot(data=metrics_df)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()

    def plot_topic_distribution(self, topic_data, title='Topic Distribution'):
        """Plot topic distribution across responses."""
        plt.figure(figsize=self.default_figsize)
        
        # Create stacked bar chart
        df = pd.DataFrame(topic_data)
        df.plot(kind='bar', stacked=True)
        plt.title(title)
        plt.xlabel('Response ID')
        plt.ylabel('Topic Probability')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return plt.gcf()

    def plot_model_comparison(self, model_scores, metric='quality'):
        """Plot comparison between different models."""
        plt.figure(figsize=self.default_figsize)
        
        # Create violin plots for model comparison
        sns.violinplot(data=model_scores)
        plt.title(f'Model Comparison - {metric.capitalize()}')
        plt.xlabel('Model')
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()

    def create_summary_dashboard(self, data_dict):
        """Create a comprehensive dashboard of visualizations."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 2)

        # Length distribution
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(data_dict['lengths'], ax=ax1)
        ax1.set_title('Response Length Distribution')

        # Quality metrics
        ax2 = fig.add_subplot(gs[0, 1])
        sns.boxplot(data=data_dict['quality_metrics'], ax=ax2)
        ax2.set_title('Quality Metrics')
        ax2.tick_params(axis='x', rotation=45)

        # Topic distribution
        ax3 = fig.add_subplot(gs[1, 0])
        pd.DataFrame(data_dict['topics']).plot(kind='bar', stacked=True, ax=ax3)
        ax3.set_title('Topic Distribution')
        ax3.legend(bbox_to_anchor=(1.05, 1))

        # Model comparison
        ax4 = fig.add_subplot(gs[1, 1])
        sns.violinplot(data=data_dict['model_scores'], ax=ax4)
        ax4.set_title('Model Comparison')
        
        plt.tight_layout()
        return fig