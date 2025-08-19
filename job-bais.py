import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class FairHiringModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.protected_attribute = 'gender'
        self.privileged_class = 1  # male
        self.unprivileged_class = 0  # female
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic hiring data with inherent bias"""
        data = []
        
        for _ in range(n_samples):
            # Random gender (0: female, 1: male)
            gender = np.random.choice([0, 1], p=[0.4, 0.6])
            
            # Experience with bias: men have slightly more experience on average
            experience = np.random.normal(5 if gender == 1 else 4.5, 2)
            experience = max(0, min(experience, 15))
            
            # Education level
            education = np.random.choice([1, 2, 3, 4], p=[0.1, 0.3, 0.4, 0.2])
            
            # Skill score (slightly biased towards men)
            skill_score = np.random.normal(70 + gender * 5, 15)
            skill_score = max(0, min(skill_score, 100))
            
            # Career gap - more common for women
            has_gap = np.random.choice([0, 1], p=[0.7 if gender == 1 else 0.3, 
                                                  0.3 if gender == 1 else 0.7])
            gap_length = np.random.exponential(1.5) if has_gap else 0
            gap_length = min(gap_length, 5)  # cap at 5 years
            
            # Relevant certifications
            certifications = np.random.poisson(1 + gender * 0.5)
            
            # Historical bias: women with gaps were less likely to be hired
            # True qualification based on skills, not gender or gap
            true_qualification = (skill_score * 0.4 + experience * 0.3 + 
                                 education * 0.2 + certifications * 0.1)
            
            # Historical biased hiring decision
            historical_bias = -10 if (gender == 0 and has_gap == 1) else 0
            hired = 1 if (true_qualification + historical_bias + 
                         np.random.normal(0, 5)) > 60 else 0
            
            data.append({
                'gender': gender,
                'experience': experience,
                'education': education,
                'skill_score': skill_score,
                'has_career_gap': has_gap,
                'career_gap_length': gap_length,
                'certifications': certifications,
                'hired': hired
            })
        
        return pd.DataFrame(data)
    
    def analyze_bias(self, df):
        """Analyze bias in the dataset"""
        print("=== BIAS ANALYSIS ===")
        
        # Hiring rate by gender
        hiring_by_gender = df.groupby('gender')['hired'].mean()
        print(f"Hiring rate - Female: {hiring_by_gender[0]:.2%}, Male: {hiring_by_gender[1]:.2%}")
        
        # Hiring rate by gender and career gap
        hiring_by_gap = df.groupby(['gender', 'has_career_gap'])['hired'].mean()
        print("\nHiring rate by gender and career gap:")
        for (gender, gap), rate in hiring_by_gap.items():
            gender_str = "Female" if gender == 0 else "Male"
            gap_str = "Has gap" if gap == 1 else "No gap"
            print(f"{gender_str} with {gap_str}: {rate:.2%}")
        
        # Disparate impact ratio (should be close to 1 for fairness)
        di_ratio = hiring_by_gender[0] / hiring_by_gender[1]
        print(f"\nDisparate Impact Ratio: {di_ratio:.3f}")
        print("(Values < 0.8 indicate potential adverse impact)")
        
        return di_ratio
    
    def preprocess_data(self, df, apply_fairness=False):
        """Preprocess data and optionally apply fairness techniques"""
        X = df.drop('hired', axis=1)
        y = df['hired']
        
        if apply_fairness:
            # Convert to AIF360 format for fairness processing
            aif_dataset = BinaryLabelDataset(
                favorable_label=1,
                unfavorable_label=0,
                df=df,
                label_names=['hired'],
                protected_attribute_names=['gender']
            )
            
            # Apply reweighing to reduce bias
            RW = Reweighing(unprivileged_groups=[{'gender': 0}],
                          privileged_groups=[{'gender': 1}])
            aif_dataset_transf = RW.fit_transform(aif_dataset)
            
            # Convert back to pandas DataFrame
            df_transf = aif_dataset_transf.convert_to_dataframe()[0]
            X = df_transf.drop('hired', axis=1)
            y = df_transf['hired']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale numerical features
        numerical_features = ['experience', 'skill_score', 'career_gap_length', 'certifications']
        X_train[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = self.scaler.transform(X_test[numerical_features])
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, use_fair_features=True):
        """Train the hiring model"""
        if use_fair_features:
            # Remove protected attribute and focus on fair features
            X_train_fair = X_train.drop(['gender'], axis=1)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_fair, y_train)
        else:
            # Use all features (including protected attribute)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
        
        self.model = model
        return model
    
    def evaluate_model(self, model, X_test, y_test, use_fair_features=True):
        """Evaluate model performance and fairness"""
        if use_fair_features:
            X_test_fair = X_test.drop(['gender'], axis=1)
            y_pred = model.predict(X_test_fair)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Analyze fairness metrics
        test_results = X_test.copy()
        test_results['actual'] = y_test
        test_results['predicted'] = y_pred
        
        # Hiring rate by gender
        hiring_rate_female = test_results[test_results['gender'] == 0]['predicted'].mean()
        hiring_rate_male = test_results[test_results['gender'] == 1]['predicted'].mean()
        
        print(f"\nPredicted Hiring Rate - Female: {hiring_rate_female:.2%}")
        print(f"Predicted Hiring Rate - Male: {hiring_rate_male:.2%}")
        
        di_ratio = hiring_rate_female / hiring_rate_male
        print(f"Disparate Impact Ratio: {di_ratio:.3f}")
        
        return test_results, di_ratio
    
    def feature_importance_analysis(self, model, X_test):
        """Analyze which features are most important"""
        X_test_fair = X_test.drop(['gender'], axis=1)
        importances = model.feature_importances_
        feature_names = X_test_fair.columns
        
        # Create feature importance dataframe
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(fi_df)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=fi_df)
        plt.title('Feature Importance for Hiring Decisions')
        plt.tight_layout()
        plt.show()
        
        return fi_df
    
    def mitigation_strategies(self, df):
        """Demonstrate various bias mitigation strategies"""
        print("\n" + "="*50)
        print("BIAS MITIGATION STRATEGIES COMPARISON")
        print("="*50)
        
        strategies = {
            'Baseline (Biased)': {'apply_fairness': False, 'use_fair_features': False},
            'Remove Gender Feature': {'apply_fairness': False, 'use_fair_features': True},
            'Reweighing + Fair Features': {'apply_fairness': True, 'use_fair_features': True}
        }
        
        results = {}
        
        for strategy, params in strategies.items():
            print(f"\n--- {strategy} ---")
            
            # Preprocess data
            X_train, X_test, y_train, y_test = self.preprocess_data(
                df, apply_fairness=params['apply_fairness']
            )
            
            # Train model
            model = self.train_model(X_train, y_train, 
                                   use_fair_features=params['use_fair_features'])
            
            # Evaluate model
            test_results, di_ratio = self.evaluate_model(
                model, X_test, y_test, 
                use_fair_features=params['use_fair_features']
            )
            
            results[strategy] = {
                'di_ratio': di_ratio,
                'female_hiring_rate': test_results[test_results['gender'] == 0]['predicted'].mean(),
                'male_hiring_rate': test_results[test_results['gender'] == 1]['predicted'].mean()
            }
        
        # Compare results
        print("\n" + "="*50)
        print("COMPARISON OF MITIGATION STRATEGIES")
        print("="*50)
        
        comparison_df = pd.DataFrame(results).T
        print(comparison_df)
        
        return comparison_df

# Main execution
if __name__ == "__main__":
    # Initialize the fair hiring model
    fair_model = FairHiringModel()
    
    # Generate synthetic data with historical bias
    print("Generating synthetic hiring data with historical bias...")
    hiring_data = fair_model.generate_synthetic_data(2000)
    
    # Analyze initial bias
    print("Initial dataset characteristics:")
    print(f"Dataset shape: {hiring_data.shape}")
    print(f"Female applicants: {(hiring_data['gender'] == 0).sum()}")
    print(f"Male applicants: {(hiring_data['gender'] == 1).sum()}")
    print(f"Overall hiring rate: {hiring_data['hired'].mean():.2%}")
    
    initial_di = fair_model.analyze_bias(hiring_data)
    
    # Compare different mitigation strategies
    results_comparison = fair_model.mitigation_strategies(hiring_data)
    
    # Train final fair model
    print("\nTraining final fair model...")
    X_train, X_test, y_train, y_test = fair_model.preprocess_data(
        hiring_data, apply_fairness=True
    )
    final_model = fair_model.train_model(X_train, y_train, use_fair_features=True)
    test_results, final_di = fair_model.evaluate_model(final_model, X_test, y_test, use_fair_features=True)
    
    # Feature importance analysis
    fi_df = fair_model.feature_importance_analysis(final_model, X_test)
    
    print(f"\nImprovement in Disparate Impact Ratio: {final_di/initial_di:.2%}")
