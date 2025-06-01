from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.dataset import load_dataset
from src.features import create_modular_pipeline
from src.globals import logger

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_survival_overview(df: pd.DataFrame, output_dir: Path):
    """Create survival overview plots"""
    logger.info("Creating survival overview plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Titanic Survival Analysis Overview', fontsize=16, fontweight='bold')
    
    # Overall survival rate
    survival_counts = df['Survived'].value_counts()
    axes[0, 0].pie(survival_counts.values, labels=['Died', 'Survived'], autopct='%1.1f%%',
                   colors=['#ff6b6b', '#4ecdc4'])
    axes[0, 0].set_title('Overall Survival Rate')
    
    # Survival by gender
    survival_gender = df.groupby(['Sex', 'Survived']).size().unstack()
    survival_gender.plot(kind='bar', ax=axes[0, 1], color=['#ff6b6b', '#4ecdc4'])
    axes[0, 1].set_title('Survival by Gender')
    axes[0, 1].set_xlabel('Gender')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend(['Died', 'Survived'])
    axes[0, 1].tick_params(axis='x', rotation=0)
    
    # Survival by class
    survival_class = df.groupby(['Pclass', 'Survived']).size().unstack()
    survival_class.plot(kind='bar', ax=axes[1, 0], color=['#ff6b6b', '#4ecdc4'])
    axes[1, 0].set_title('Survival by Passenger Class')
    axes[1, 0].set_xlabel('Passenger Class')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend(['Died', 'Survived'])
    axes[1, 0].tick_params(axis='x', rotation=0)
    
    # Survival by age group
    if 'AgeBin' in df.columns:
        age_survival = df.groupby(['AgeBin', 'Survived']).size().unstack(fill_value=0)
        age_survival.plot(kind='bar', ax=axes[1, 1], color=['#ff6b6b', '#4ecdc4'])
        axes[1, 1].set_title('Survival by Age Group')
        axes[1, 1].set_xlabel('Age Group')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend(['Died', 'Survived'])
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'survival_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_demographic_analysis(df: pd.DataFrame, output_dir: Path):
    """Create demographic analysis plots"""
    logger.info("Creating demographic analysis plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Demographic Analysis', fontsize=16, fontweight='bold')
    
    # Age distribution
    df['Age'].hist(bins=30, ax=axes[0, 0], alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    
    # Age by survival
    survived = df[df['Survived'] == 1]['Age'].dropna()
    died = df[df['Survived'] == 0]['Age'].dropna()
    axes[0, 1].hist([died, survived], bins=30, alpha=0.7, 
                    label=['Died', 'Survived'], color=['#ff6b6b', '#4ecdc4'])
    axes[0, 1].set_title('Age Distribution by Survival')
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Fare distribution
    df['Fare'].hist(bins=30, ax=axes[0, 2], alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 2].set_title('Fare Distribution')
    axes[0, 2].set_xlabel('Fare')
    axes[0, 2].set_ylabel('Frequency')
    
    # Family size distribution
    df['FamilySize'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0], color='gold')
    axes[1, 0].set_title('Family Size Distribution')
    axes[1, 0].set_xlabel('Family Size')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=0)
    
    # Embarked distribution
    df['Embarked'].value_counts().plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%')
    axes[1, 1].set_title('Embarked Port Distribution')
    
    # Class distribution
    df['Pclass'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 2], color='mediumpurple')
    axes[1, 2].set_title('Passenger Class Distribution')
    axes[1, 2].set_xlabel('Class')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'demographic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_analysis(df: pd.DataFrame, output_dir: Path):
    """Create correlation analysis plots"""
    logger.info("Creating correlation analysis...")
    
    # Select numeric columns for correlation
    numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
    df_numeric = df[numeric_cols].dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
    
    # Correlation heatmap
    correlation_matrix = df_numeric.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, ax=axes[0])
    axes[0].set_title('Feature Correlation Matrix')
    
    # Survival correlation
    survival_corr = correlation_matrix['Survived'].drop('Survived').sort_values(key=abs, ascending=False)
    survival_corr.plot(kind='bar', ax=axes[1], color=['red' if x < 0 else 'green' for x in survival_corr])
    axes[1].set_title('Feature Correlation with Survival')
    axes[1].set_xlabel('Features')
    axes[1].set_ylabel('Correlation with Survival')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_survival_rates(df: pd.DataFrame, output_dir: Path):
    """Create detailed survival rate analysis"""
    logger.info("Creating survival rate analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Survival Rates by Different Factors', fontsize=16, fontweight='bold')
    
    # Survival rate by gender and class
    survival_rate_gender_class = df.groupby(['Sex', 'Pclass'])['Survived'].mean().unstack()
    survival_rate_gender_class.plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Survival Rate by Gender and Class')
    axes[0, 0].set_xlabel('Gender')
    axes[0, 0].set_ylabel('Survival Rate')
    axes[0, 0].legend(title='Class')
    axes[0, 0].tick_params(axis='x', rotation=0)
    
    # Survival rate by age group and gender
    if 'AgeBin' in df.columns:
        age_gender_survival = df.groupby(['AgeBin', 'Sex'])['Survived'].mean().unstack()
        age_gender_survival.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Survival Rate by Age Group and Gender')
        axes[0, 1].set_xlabel('Age Group')
        axes[0, 1].set_ylabel('Survival Rate')
        axes[0, 1].legend(title='Gender')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Survival rate by family size
    family_survival = df.groupby('FamilySize')['Survived'].agg(['mean', 'count'])
    family_survival['mean'].plot(kind='bar', ax=axes[1, 0], color='orange')
    axes[1, 0].set_title('Survival Rate by Family Size')
    axes[1, 0].set_xlabel('Family Size')
    axes[1, 0].set_ylabel('Survival Rate')
    axes[1, 0].tick_params(axis='x', rotation=0)
    
    # Add count as text on bars
    for i, (idx, row) in enumerate(family_survival.iterrows()):
        axes[1, 0].text(i, row['mean'] + 0.01, f"n={row['count']}", 
                       ha='center', va='bottom', fontsize=8)
    
    # Survival rate by embarked port
    embarked_survival = df.groupby('Embarked')['Survived'].mean()
    embarked_survival.plot(kind='bar', ax=axes[1, 1], color='teal')
    axes[1, 1].set_title('Survival Rate by Embarked Port')
    axes[1, 1].set_xlabel('Embarked Port')
    axes[1, 1].set_ylabel('Survival Rate')
    axes[1, 1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'survival_rates.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_missing_data_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze and visualize missing data patterns"""
    logger.info("Creating missing data analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Missing Data Analysis', fontsize=16, fontweight='bold')
    
    # Missing data by column
    missing_data = df.isnull().sum().sort_values(ascending=False)
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) > 0:
        missing_data.plot(kind='bar', ax=axes[0], color='coral')
        axes[0].set_title('Missing Data Count by Column')
        axes[0].set_xlabel('Columns')
        axes[0].set_ylabel('Missing Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Missing data percentage
        missing_percentage = (missing_data / len(df)) * 100
        missing_percentage.plot(kind='bar', ax=axes[1], color='lightcoral')
        axes[1].set_title('Missing Data Percentage by Column')
        axes[1].set_xlabel('Columns')
        axes[1].set_ylabel('Missing Percentage (%)')
        axes[1].tick_params(axis='x', rotation=45)
    else:
        axes[0].text(0.5, 0.5, 'No Missing Data Found', ha='center', va='center', transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, 'No Missing Data Found', ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'missing_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_fare_analysis(df: pd.DataFrame, output_dir: Path):
    """Detailed fare analysis"""
    logger.info("Creating fare analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fare Analysis', fontsize=16, fontweight='bold')
    
    # Fare by class boxplot
    df.boxplot(column='Fare', by='Pclass', ax=axes[0, 0])
    axes[0, 0].set_title('Fare Distribution by Class')
    axes[0, 0].set_xlabel('Passenger Class')
    axes[0, 0].set_ylabel('Fare')
    
    # Fare by survival
    df.boxplot(column='Fare', by='Survived', ax=axes[0, 1])
    axes[0, 1].set_title('Fare Distribution by Survival')
    axes[0, 1].set_xlabel('Survived (0=No, 1=Yes)')
    axes[0, 1].set_ylabel('Fare')
    
    # Fare vs Age scatter plot
    survived_passengers = df[df['Survived'] == 1]
    died_passengers = df[df['Survived'] == 0]
    
    axes[1, 0].scatter(died_passengers['Age'], died_passengers['Fare'], 
                      alpha=0.6, c='red', label='Died', s=20)
    axes[1, 0].scatter(survived_passengers['Age'], survived_passengers['Fare'], 
                      alpha=0.6, c='green', label='Survived', s=20)
    axes[1, 0].set_title('Fare vs Age by Survival')
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Fare')
    axes[1, 0].legend()
    
    # Average fare by embarked port
    fare_embarked = df.groupby('Embarked')['Fare'].mean()
    fare_embarked.plot(kind='bar', ax=axes[1, 1], color='purple')
    axes[1, 1].set_title('Average Fare by Embarked Port')
    axes[1, 1].set_xlabel('Embarked Port')
    axes[1, 1].set_ylabel('Average Fare')
    axes[1, 1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fare_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_statistics(df: pd.DataFrame, output_dir: Path):
    """Generate and save summary statistics"""
    logger.info("Generating summary statistics...")
    
    # Basic statistics
    summary_stats = df.describe()
    
    # Survival statistics
    survival_stats = {
        'Overall Survival Rate': df['Survived'].mean(),
        'Male Survival Rate': df[df['Sex'] == 'male']['Survived'].mean(),
        'Female Survival Rate': df[df['Sex'] == 'female']['Survived'].mean(),
        'Class 1 Survival Rate': df[df['Pclass'] == 1]['Survived'].mean(),
        'Class 2 Survival Rate': df[df['Pclass'] == 2]['Survived'].mean(),
        'Class 3 Survival Rate': df[df['Pclass'] == 3]['Survived'].mean(),
    }
    
    # Save to text file
    with open(output_dir / 'summary_statistics.txt', 'w') as f:
        f.write("TITANIC DATASET - EXPLORATORY DATA ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("DATASET OVERVIEW:\n")
        f.write(f"Total passengers: {len(df)}\n")
        f.write(f"Total features: {len(df.columns)}\n")
        f.write(f"Missing values: {df.isnull().sum().sum()}\n\n")
        
        f.write("SURVIVAL STATISTICS:\n")
        for key, value in survival_stats.items():
            f.write(f"{key}: {value:.2%}\n")
        f.write("\n")
        
        f.write("BASIC STATISTICS:\n")
        f.write(summary_stats.to_string())
        f.write("\n\n")
        
        f.write("MISSING DATA BY COLUMN:\n")
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        for col, count in missing_data.items():
            f.write(f"{col}: {count} ({count/len(df):.1%})\n")

@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig):
    """
    Generate comprehensive EDA plots for Titanic dataset using Hydra configuration
    """
    
    # Get paths from Hydra config
    proj_root = Path(hydra.utils.get_original_cwd())
    raw_data_path = proj_root / cfg["paths"]["data"]["raw_data"] / cfg["names"]["train_data"]
    figures_dir = proj_root / cfg["paths"]["reports_parent_dir"] / cfg["paths"]["figures_parent_dir"]
    
    # Create output directory if it doesn't exist
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Titanic EDA analysis...")
    logger.info(f"Using configuration from: {cfg}")
    
    # Load data and apply feature engineering
    df = load_dataset(raw_data_path)
    feature_engineer = create_modular_pipeline()
    df = feature_engineer.fit_transform(df)
    logger.info(df.columns)
    
    # Generate all plots with progress tracking
    plot_functions = [
        ("Survival Overview", plot_survival_overview),
        ("Demographic Analysis", plot_demographic_analysis), 
        ("Correlation Analysis", plot_correlation_analysis),
        ("Survival Rates", plot_survival_rates),
        ("Missing Data Analysis", plot_missing_data_analysis),
        ("Fare Analysis", plot_fare_analysis),
    ]
    
    for description, plot_func in tqdm(plot_functions, desc="Generating plots"):
        logger.info(f"Generating {description}...")
        plot_func(df, figures_dir)
        
    # Generate summary statistics
    generate_summary_statistics(df, figures_dir)
    
    logger.success(f"EDA analysis complete! All plots saved to {figures_dir}")
    logger.info(f"Generated {len(plot_functions)} visualization files:")
    logger.info("- survival_overview.png")
    logger.info("- demographic_analysis.png") 
    logger.info("- correlation_analysis.png")
    logger.info("- survival_rates.png")
    logger.info("- missing_data_analysis.png")
    logger.info("- fare_analysis.png")
    logger.info("- summary_statistics.txt")


if __name__ == "__main__":
    # Run with Hydra configuration by default
    main()
