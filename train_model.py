"""
Integrated ML Model Training Pipeline for Network Intrusion Detection
Combines preprocessing, feature engineering, model training, and ensemble stacking
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import yaml
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
import xgboost as xgb


class DataPreprocessor:
    """Handles data preprocessing and sampling"""
    
    def __init__(self, config: dict):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.label_column = None
        self.label_mapping = {}  # Initialize empty label mapping
        
    def preprocess_data(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
        """
        Preprocess data: normalization, encoding, sampling
        
        Args:
            df: Input dataframe
            fit: Whether to fit label encoder (True for training)
            
        Returns:
            Processed features, labels, label encoder
        """
        print("📊 Preprocessing data...")
        
        # Find label column first
        label_col = self._find_label_column(df)
        if not label_col:
            raise ValueError("No label column found in dataset")
        
        # Store label column name
        if fit:
            self.label_column = label_col
        
        # Identify feature columns (non-object types, excluding label)
        if self.feature_columns is None or fit:
            # Get all numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            # Remove label column
            self.feature_columns = [col for col in numeric_cols if col != label_col]
            
            # If we have very few numeric columns, also include encoded categorical
            if len(self.feature_columns) < 5:
                print(f"  ⚠ Only {len(self.feature_columns)} numeric columns found")
                print(f"  ℹ Including all non-label columns as features")
                self.feature_columns = [col for col in df.columns if col != label_col]
        
        print(f"  ✓ Found {len(self.feature_columns)} feature columns")
        
        # Handle missing features in new data
        if not fit:
            missing_cols = set(self.feature_columns) - set(df.columns)
            if missing_cols:
                print(f"  ⚠ Warning: {len(missing_cols)} expected columns missing from input")
                print(f"  ℹ Adding missing columns with zeros")
                for col in missing_cols:
                    df[col] = 0
            
            # Select only the columns we need
            available_features = [col for col in self.feature_columns if col in df.columns]
            if len(available_features) < len(self.feature_columns):
                print(f"  ⚠ Using {len(available_features)}/{len(self.feature_columns)} features")
        
        # Get feature data
        try:
            feature_data = df[self.feature_columns].copy()
        except KeyError as e:
            print(f"  ✗ Error: Missing columns: {e}")
            print(f"  ℹ Available columns: {list(df.columns)}")
            print(f"  ℹ Expected features: {self.feature_columns[:10]}...")
            raise ValueError(f"Dataset missing required columns. Train and test data must have same columns.")
        
        # Z-score normalization on numeric columns only
        numeric_features = feature_data.select_dtypes(include=['number']).columns
        if len(numeric_features) > 0:
            print("  ↳ Applying Z-score normalization...")
            feature_data[numeric_features] = feature_data[numeric_features].apply(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
        
        # Fill missing values
        feature_data = feature_data.fillna(0)
        
        # Encode labels
        if fit:
            y = self.label_encoder.fit_transform(df[label_col])
            
            # Create label mapping during fit
            self.label_mapping = {
                i: label for i, label in enumerate(self.label_encoder.classes_)
            }
            
            print(f"  ✓ Encoded labels: {sorted(set(y))}")
            print(f"  ✓ Label mapping created: {self.label_mapping}")
        else:
            try:
                y = self.label_encoder.transform(df[label_col])
                print(f"  ✓ Encoded labels: {sorted(set(y))}")
            except ValueError as e:
                print(f"  ⚠ Warning: Unknown labels in test data")
                print(f"  ℹ Known classes: {list(self.label_encoder.classes_)}")
                print(f"  ℹ Test data has: {list(df[label_col].unique())}")
                
                # Handle unknown labels gracefully
                # Map unknown labels to -1 or nearest known class
                print(f"  ℹ Mapping unknown labels to known classes...")
                
                # Create a safe mapping
                known_classes = set(self.label_encoder.classes_)
                test_labels = df[label_col].values
                
                # For each test label, map to known or encode as new
                safe_labels = []
                for label in test_labels:
                    if label in known_classes:
                        safe_labels.append(label)
                    else:
                        # Use first known class as fallback
                        safe_labels.append(list(known_classes)[0])
                        print(f"    • Mapped '{label}' → '{list(known_classes)[0]}'")
                
                y = self.label_encoder.transform(safe_labels)
        
        return feature_data, y, self.label_encoder
    
    def _find_label_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the label column in dataframe"""
        candidates = ['Label', 'label', 'attack_cat', 'class', 'Class', 'attack_type', 'threat', 'category']
        for col in candidates:
            if col in df.columns:
                return col
        
        # If no standard name found, check for columns with 'label' or 'class' in name
        for col in df.columns:
            if 'label' in col.lower() or 'class' in col.lower() or 'attack' in col.lower():
                return col
        
        return None
    
    def apply_sampling(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply MiniBatchKMeans-based sampling for imbalanced data
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Sampled features and labels
        """
        print("🎲 Applying data sampling...")
        
        # Check class distribution BEFORE sampling
        initial_counts = pd.Series(y).value_counts()
        print(f"\n  Initial class distribution:")
        for class_idx, count in sorted(initial_counts.items()):
            # Use label_mapping if it exists, otherwise just show class index
            class_name = self.label_mapping.get(class_idx, f"Class {class_idx}")
            print(f"    {class_name}: {count} samples")
        
        # Check if we have multiple classes
        if len(initial_counts) < 2:
            print(f"\n  ⚠ WARNING: Only {len(initial_counts)} class in data!")
            print(f"  ℹ Skipping sampling - need at least 2 classes for meaningful training")
            return X, y
        
        df_temp = X.copy()
        df_temp['Label'] = y
        
        # Separate minority and majority classes
        minority_labels = self.config['sampling'].get('minority_labels', [])
        
        # Filter minority classes that exist in data
        minority_labels = [l for l in minority_labels if l in initial_counts.index]
        
        if not minority_labels:
            print("  ℹ No minority classes specified or found, skipping sampling")
            return X, y
        
        df_minor = df_temp[df_temp['Label'].isin(minority_labels)]
        df_major = df_temp.drop(df_minor.index)
        
        if len(df_major) == 0:
            print("  ℹ No majority class samples, skipping sampling")
            return X, y
        
        # Check if majority class has enough samples for clustering
        if len(df_major) < 10:
            print(f"  ⚠ Majority class too small ({len(df_major)} samples), skipping sampling")
            return X, y
        
        # Apply KMeans clustering on majority class
        X_major = df_major.drop(['Label'], axis=1)
        y_major = df_major['Label'].values
        
        # Adjust cluster count based on data size
        n_clusters = min(
            self.config['sampling']['kmeans_clusters'],
            len(X_major),
            max(10, len(X_major) // 10)  # At least 10 clusters or 1/10 of data
        )
        
        print(f"  ✓ Clustering majority class with {n_clusters} clusters...")
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=self.config['sampling']['random_state'],
            batch_size=min(100, len(X_major))
        ).fit(X_major)
        
        df_major['klabel'] = kmeans.labels_
        
        # Sample from each cluster
        frac = self.config['sampling']['sampling_fraction']
        sampled_major = df_major.groupby('klabel', group_keys=False).apply(
            lambda x: x.sample(frac=min(frac, 1.0), random_state=42) if len(x) > 0 else x
        )
        
        sampled_major = sampled_major.drop(['klabel'], axis=1)
        
        # Combine sampled majority with minority
        result = pd.concat([sampled_major, df_minor], ignore_index=True)
        
        X_sampled = result.drop(['Label'], axis=1)
        y_sampled = result['Label'].values
        
        # Check class distribution AFTER sampling
        final_counts = pd.Series(y_sampled).value_counts()
        print(f"\n  After sampling:")
        for class_idx, count in sorted(final_counts.items()):
            # Use label_mapping if it exists, otherwise just show class index
            class_name = self.label_mapping.get(class_idx, f"Class {class_idx}")
            old_count = initial_counts.get(class_idx, 0)
            print(f"    {class_name}: {old_count} → {count} samples")
        
        # Verify we still have multiple classes
        if len(final_counts) < 2:
            print(f"\n  ⚠ WARNING: Sampling reduced to only {len(final_counts)} class!")
            print(f"  ℹ Reverting to original data")
            return X, y
        
        print(f"\n  ✓ Sampled from {len(df_temp)} to {len(result)} records")
        
        return X_sampled, y_sampled


class FeatureEngineer:
    """Feature selection using mutual information and FCBF"""
    
    def __init__(self, config: dict):
        self.config = config
        self.selected_features = None
        
    def select_features(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:

        print("🔍 Selecting features...")
        
        # Mutual information feature selection
        print("  ↳ Computing mutual information...")
        importances = mutual_info_classif(X, y)
        
        # Calculate feature importance
        f_list = sorted(
            zip(map(lambda x: round(x, 4), importances), X.columns),
            reverse=True
        )
        
        # Select features until 90% cumulative importance
        total_importance = sum([f[0] for f in f_list])
        cumulative = 0
        selected = []
        
        threshold = self.config['feature_selection']['mutual_info_threshold']
        
        for importance, feature in f_list:
            cumulative += importance / total_importance
            selected.append(feature)
            if cumulative >= threshold:
                break
        
        print(f"  ↳ Selected {len(selected)} features (MI threshold: {threshold})")
        
        X_selected = X[selected]
        
        # Apply FCBF if available
        try:
            from FCBF_module import FCBFK
            
            print("  ↳ Applying FCBF filter...")
            k = min(self.config['feature_selection']['fcbf_k'], len(selected))
            fcbf = FCBFK(k=k)
            X_fcbf = fcbf.fit_transform(X_selected.values, y)
            
            # CRITICAL FIX: Get the ACTUAL selected feature indices from FCBF
            # FCBF has a .idx_ attribute that tells us which features were selected
            if hasattr(fcbf, 'idx_'):
                # Get the feature indices that FCBF actually selected
                selected_indices = fcbf.idx_
                # Map these indices back to the original feature names
                final_feature_names = [selected[i] for i in selected_indices]
            elif hasattr(fcbf, 'selected_features_'):
                # Alternative attribute name
                selected_indices = fcbf.selected_features_
                final_feature_names = [selected[i] for i in selected_indices]
            else:
                # Fallback: assume FCBF selected first k features in order
                print("  ⚠ FCBF doesn't expose selected indices, assuming first k features")
                final_feature_names = selected[:X_fcbf.shape[1]]
            
            # Store the actual column NAMES (not indices!)
            self.selected_features = final_feature_names
            
            X_final = pd.DataFrame(X_fcbf, columns=final_feature_names)
            
            print(f"  ✓ Final features after FCBF: {X_fcbf.shape[1]}")
            print(f"  ✓ Selected feature names: {final_feature_names}")
            
        except ImportError:
            print("  ⚠ FCBF module not available, using MI features only")
            X_final = X_selected
            # Store the column names from MI selection
            self.selected_features = selected
            print(f"  ✓ Stored {len(selected)} feature names")
            print(f"  ✓ Feature names: {selected}")
        
        # CRITICAL: Ensure selected_features is a list of column names
        if not isinstance(self.selected_features, list):
            self.selected_features = list(self.selected_features)
        
        # FINAL VERIFICATION: Print what we're storing
        print(f"\n  📋 FINAL FEATURE LIST TO BE SAVED:")
        print(f"     Count: {len(self.selected_features)}")
        print(f"     Names: {self.selected_features}")
    
        return X_final



class ModelTrainer:
    """Train and optimize ML models"""
    
    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        
    def train_base_models(self, X_train, X_test, y_train, y_test) -> dict:
        """
        Train all base models with hyperparameter optimization
        
        Returns:
            Dictionary of trained models and predictions
        """
        print("🤖 Training base models...")
        
        results = {}
        
        # XGBoost
        print("\n  [1/4] Training XGBoost...")
        xg = self._train_xgboost(X_train, X_test, y_train, y_test)
        results['xgboost'] = {
            'model': xg,
            'train_pred': xg.predict(X_train),
            'test_pred': xg.predict(X_test)
        }
        
        # Random Forest
        print("\n  [2/4] Training Random Forest...")
        rf = self._train_random_forest(X_train, X_test, y_train, y_test)
        results['random_forest'] = {
            'model': rf,
            'train_pred': rf.predict(X_train),
            'test_pred': rf.predict(X_test)
        }
        
        # Decision Tree
        print("\n  [3/4] Training Decision Tree...")
        dt = self._train_decision_tree(X_train, X_test, y_train, y_test)
        results['decision_tree'] = {
            'model': dt,
            'train_pred': dt.predict(X_train),
            'test_pred': dt.predict(X_test)
        }
        
        # Extra Trees
        print("\n  [4/4] Training Extra Trees...")
        et = self._train_extra_trees(X_train, X_test, y_train, y_test)
        results['extra_trees'] = {
            'model': et,
            'train_pred': et.predict(X_train),
            'test_pred': et.predict(X_test)
        }
        
        return results
    
    def _train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost with optimized hyperparameters"""
        params = self.config['models']['xgboost']
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        acc = model.score(X_test, y_test)
        print(f"    ✓ XGBoost Accuracy: {acc:.4f}")
        
        return model
    
    def _train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest with optimized hyperparameters"""
        params = self.config['models']['random_forest']
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        acc = model.score(X_test, y_test)
        print(f"    ✓ Random Forest Accuracy: {acc:.4f}")
        
        return model
    
    def _train_decision_tree(self, X_train, X_test, y_train, y_test):
        """Train Decision Tree with optimized hyperparameters"""
        params = self.config['models']['decision_tree']
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        
        acc = model.score(X_test, y_test)
        print(f"    ✓ Decision Tree Accuracy: {acc:.4f}")
        
        return model
    
    def _train_extra_trees(self, X_train, X_test, y_train, y_test):
        """Train Extra Trees with optimized hyperparameters"""
        params = self.config['models']['extra_trees']
        model = ExtraTreesClassifier(**params)
        model.fit(X_train, y_train)
        
        acc = model.score(X_test, y_test)
        print(f"    ✓ Extra Trees Accuracy: {acc:.4f}")
        
        return model
    
    def train_stacking_ensemble(self, base_results, y_train, y_test):
        """
        Train stacking ensemble meta-learner
        
        Args:
            base_results: Dictionary of base model results
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            Trained stacking model
        """
        print("\n🎯 Training Stacking Ensemble...")
        
        # Prepare meta-features
        train_meta = np.column_stack([
            base_results['decision_tree']['train_pred'],
            base_results['extra_trees']['train_pred'],
            base_results['random_forest']['train_pred'],
            base_results['xgboost']['train_pred']
        ])
        
        test_meta = np.column_stack([
            base_results['decision_tree']['test_pred'],
            base_results['extra_trees']['test_pred'],
            base_results['random_forest']['test_pred'],
            base_results['xgboost']['test_pred']
        ])
        
        # Train meta-learner
        params = self.config['models']['stacking_meta']
        meta_model = xgb.XGBClassifier(**params)
        meta_model.fit(train_meta, y_train)
        
        # Evaluate
        acc = meta_model.score(test_meta, y_test)
        y_pred = meta_model.predict(test_meta)
        
        print(f"  ✓ Stacking Accuracy: {acc:.4f}")
        
        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        print(f"  ✓ Precision: {precision:.4f}")
        print(f"  ✓ Recall: {recall:.4f}")
        print(f"  ✓ F1-Score: {fscore:.4f}")
        
        return meta_model, acc


class ThreatModelTrainer:
    """Complete training pipeline"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessor = DataPreprocessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.final_model = None
        self.label_mapping = {}
        
    def train(self, data_path: str) -> float:
        """
        Complete training pipeline
        
        Args:
            data_path: Path to training data CSV
            
        Returns:
            Final model accuracy
        """
        print("\n" + "="*70)
        print("🚀 STARTING MODEL TRAINING PIPELINE")
        print("="*70 + "\n")
        
        # Load data
        print(f"📂 Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"  ✓ Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Filter to keep only 2 classes (automatically detects which classes exist)
        print(f"\n🔍 Filtering classes to keep only 2 classes...")
        label_col = self.preprocessor._find_label_column(df)
        if not label_col:
            raise ValueError("No label column found in dataset")
        
        print(f"  Label column: {label_col}")
        print(f"  Classes in data:")
        
        # Get all unique classes sorted by count (most common first)
        class_counts = df[label_col].value_counts()
        all_classes = sorted(class_counts.index)
        
        for label, count in class_counts.items():
            percentage = (count / len(df)) * 100
            print(f"    - {label}: {count} samples ({percentage:.1f}%)")
        
        # Determine which 2 classes to keep
        # Strategy: Keep the 2 most common classes
        classes_to_keep = class_counts.head(2).index.tolist()
        
        print(f"\n  📌 Strategy: Keeping the 2 most common classes")
        print(f"  ✓ Selected classes to keep: {classes_to_keep}")
        
        # Alternative: If you want to keep first 2 by value, uncomment this:
        # classes_to_keep = sorted(all_classes)[:2]
        # print(f"  ✓ Selected classes to keep (first 2 by value): {classes_to_keep}")
        
        # Filter data
        original_size = len(df)
        df = df[df[label_col].isin(classes_to_keep)].copy()
        filtered_size = len(df)
        removed_count = original_size - filtered_size
        
        print(f"\n  ✓ Removed {removed_count} records from other classes")
        print(f"  ✓ Remaining: {filtered_size} records")
        
        if filtered_size == 0:
            raise ValueError(f"No records found! Available classes were: {all_classes}")
        
        print(f"\n  Final class distribution:")
        for label, count in df[label_col].value_counts().items():
            percentage = (count / filtered_size) * 100
            print(f"    - {label}: {count} samples ({percentage:.1f}%)")
        
        # Preprocess
        X, y, label_encoder = self.preprocessor.preprocess_data(df, fit=True)
        
        # Store label mapping from preprocessor
        self.label_mapping = self.preprocessor.label_mapping
        
        # Create label mapping for display (also store in preprocessor)
        print(f"\n📋 Label Mapping:")
        for idx, label in self.label_mapping.items():
            count = np.sum(y == idx)
            print(f"  {idx}: {label} ({count} samples)")
        
        # Apply sampling if configured
        if self.config.get('sampling', {}).get('kmeans_clusters'):
            X, y = self.preprocessor.apply_sampling(X, y)
        
        # Feature engineering
        X_selected = self.feature_engineer.select_features(X, y)
        
        # Train-test split
        print(f"\n✂️  Splitting data...")
        split_config = self.config['split']
        
        # Check we have enough samples per class for stratified split
        class_counts = pd.Series(y).value_counts()
        min_class_count = class_counts.min()
        
        if min_class_count < 2:
            print(f"  ⚠ WARNING: Smallest class has only {min_class_count} sample(s)")
            print(f"  ℹ Cannot use stratified split - using random split")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y,
                train_size=split_config['train_size'],
                test_size=split_config['test_size'],
                random_state=split_config['random_state'],
                stratify=None  # Don't stratify if not enough samples
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y,
                train_size=split_config['train_size'],
                test_size=split_config['test_size'],
                random_state=split_config['random_state'],
                stratify=y if split_config['stratify'] else None
            )
        
        print(f"  ✓ Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Verify both train and test have multiple classes
        train_classes = len(pd.Series(y_train).unique())
        test_classes = len(pd.Series(y_test).unique())
        
        print(f"  ✓ Train has {train_classes} classes, Test has {test_classes} classes")
        
        if train_classes < 2:
            print(f"\n  ❌ ERROR: Training data has only {train_classes} class!")
            print(f"  ℹ This usually means:")
            print(f"     • Data sampling was too aggressive")
            print(f"     • Not enough data for train/test split")
            print(f"     • Data filtering removed too many samples")
            print(f"\n  SOLUTIONS:")
            print(f"     1. Disable sampling in config.yaml (comment out 'sampling' section)")
            print(f"     2. Increase sampling_fraction (e.g., from 0.008 to 0.1)")
            print(f"     3. Use larger dataset")
            print(f"     4. Adjust train/test split (e.g., 90/10 instead of 80/20)")
            raise ValueError(f"Cannot train model with only {train_classes} class in training data")
        
        if test_classes < 2:
            print(f"  ⚠ WARNING: Test data has only {test_classes} class")
            print(f"  ℹ Evaluation metrics may be limited")
        
        # Apply SMOTE if configured
        if 'smote' in self.config:
            print(f"\n⚖️  Applying SMOTE...")
            smote_config = self.config['smote']
            
            # Check class distribution
            class_counts = pd.Series(y_train).value_counts()
            print(f"\n  Current class distribution:")
            for class_idx, count in sorted(class_counts.items()):
                class_name = self.label_mapping.get(class_idx, f"Class {class_idx}")
                print(f"    {class_name}: {count} samples")
            
            # Build sampling strategy for existing classes
            # Only oversample classes that have fewer samples than target
            strategy = {}
            for class_label, target_count in smote_config['sampling_strategy'].items():
                class_idx = int(class_label.split('_')[1])
                
                if class_idx in class_counts:
                    current_count = class_counts[class_idx]
                    
                    # SMOTE can only oversample, not undersample
                    if current_count < target_count:
                        strategy[class_idx] = target_count
                        print(f"  ✓ Will oversample class {class_idx}: {current_count} → {target_count}")
                    else:
                        print(f"  ℹ Skipping class {class_idx}: already has {current_count} samples (target: {target_count})")
                else:
                    print(f"  ⚠ Class {class_idx} not found in training data")
            
            if strategy:
                # Apply SMOTE only if there are classes to oversample
                print(f"\n  Applying SMOTE to {len(strategy)} class(es)...")
                smote = SMOTE(
                    sampling_strategy=strategy,
                    random_state=smote_config.get('random_state', 42)
                )
                X_train, y_train = smote.fit_resample(X_train, y_train)
                
                # Show new distribution
                new_counts = pd.Series(y_train).value_counts()
                print(f"\n  After SMOTE:")
                for class_idx, count in sorted(new_counts.items()):
                    class_name = self.label_mapping.get(class_idx, f"Class {class_idx}")
                    old_count = class_counts.get(class_idx, 0)
                    print(f"    {class_name}: {old_count} → {count} samples")
                
                print(f"\n  ✓ Total samples after SMOTE: {len(X_train)}")
            else:
                print(f"\n  ℹ No classes need oversampling - skipping SMOTE")
        
        # Train base models
        base_results = self.model_trainer.train_base_models(
            X_train, X_test, y_train, y_test
        )
        
        # Train stacking ensemble
        self.final_model, accuracy = self.model_trainer.train_stacking_ensemble(
            base_results, y_train, y_test
        )
        
        # Store all components
        self.base_models = base_results
        
        print("\n" + "="*70)
        print(f"✅ TRAINING COMPLETE - Final Accuracy: {accuracy:.4f}")
        print("="*70 + "\n")
        
        return accuracy
    
    def save_model(self, models_dir: str = "models") -> bool:
        """
        Save trained models and preprocessor
        
        Args:
            models_dir: Directory to save models
            
        Returns:
            True if successful
        """
        try:
            Path(models_dir).mkdir(exist_ok=True, parents=True)
            
            print(f"💾 Saving models to {models_dir}/...")
            
            # Save preprocessor
            joblib.dump(self.preprocessor, f"{models_dir}/preprocessor.pkl")
            print("  ✓ Saved preprocessor")
            
            # Save feature engineer
            joblib.dump(self.feature_engineer, f"{models_dir}/feature_engineer.pkl")
            print("  ✓ Saved feature engineer")
            
            # Save final stacking model
            joblib.dump(self.final_model, f"{models_dir}/threat_classifier.pkl")
            print("  ✓ Saved threat classifier")
            
            # Save label mapping
            joblib.dump(self.label_mapping, f"{models_dir}/label_mapping.pkl")
            print("  ✓ Saved label mapping")
            
            # ============ CRITICAL FIX HERE ============
            # Save the ACTUAL selected feature column names
            # NOT the feature_engineer object attributes!
            
            # Get the selected feature names from the feature_engineer
            if hasattr(self.feature_engineer, 'selected_features'):
                selected_features = self.feature_engineer.selected_features
                
                # Check what type it is
                if isinstance(selected_features, (list, tuple)):
                    # Already a list of column names - perfect!
                    feature_names_to_save = list(selected_features)
                elif isinstance(selected_features, pd.Index):
                    # Pandas Index - convert to list
                    feature_names_to_save = selected_features.tolist()
                elif isinstance(selected_features, np.ndarray):
                    if selected_features.dtype == bool:
                        # Boolean mask - need to get column names from preprocessor
                        if hasattr(self.preprocessor, 'feature_columns'):
                            all_features = self.preprocessor.feature_columns
                            feature_names_to_save = [all_features[i] for i in range(len(selected_features)) if selected_features[i]]
                        else:
                            print("  ⚠ Cannot resolve boolean mask - saving all preprocessor features")
                            feature_names_to_save = self.preprocessor.feature_columns
                    else:
                        # Array of indices or names
                        feature_names_to_save = selected_features.tolist()
                else:
                    # Unknown type - fall back to preprocessor features
                    print(f"  ⚠ Unknown selected_features type: {type(selected_features)}")
                    feature_names_to_save = self.preprocessor.feature_columns
                
                # CRITICAL VERIFICATION: Check if feature count matches model
                if hasattr(self.final_model, 'n_features_in_'):
                    expected_features = self.final_model.n_features_in_
                    actual_features = len(feature_names_to_save)
                    
                    print(f"\n  🔍 VERIFICATION:")
                    print(f"     Model expects: {expected_features} features")
                    print(f"     Saving: {actual_features} features")
                    
                    if expected_features != actual_features:
                        print(f"\n  ⚠️  WARNING: FEATURE COUNT MISMATCH!")
                        print(f"     Model was trained on {expected_features} features")
                        print(f"     But trying to save {actual_features} feature names")
                        print(f"\n     This will cause prediction errors!")
                        print(f"\n     CORRECTING: Saving only first {expected_features} features")
                        
                        # Truncate to match model expectations
                        feature_names_to_save = feature_names_to_save[:expected_features]
                
                # Save the feature names
                joblib.dump(feature_names_to_save, f"{models_dir}/selected_features.pkl")
                print(f"  ✓ Saved {len(feature_names_to_save)} selected feature names")
                
                # DEBUG: Print feature names
                print(f"    Feature names: {feature_names_to_save}")
                
            else:
                # No selected_features attribute - save all preprocessor features
                print("  ⚠ No selected_features found - saving all feature columns")
                feature_names_to_save = self.preprocessor.feature_columns
                joblib.dump(feature_names_to_save, f"{models_dir}/selected_features.pkl")
                print(f"  ✓ Saved {len(feature_names_to_save)} feature names (all features)")
            
            # ============ END FIX ============
            
            # Save base models
            for name, result in self.base_models.items():
                joblib.dump(result['model'], f"{models_dir}/{name}.pkl")
            print("  ✓ Saved base models")
            
            print("\n✅ All models saved successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ Error saving models: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Network Threat Detection Model")
    parser.add_argument('data_path', help='Path to training data CSV')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--output', default='models', help='Output directory for models')
    
    args = parser.parse_args()
    
    # Check if config exists, create default if not
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"⚠ Config file not found: {config_path}")
        print(f"  Creating default config...")
        
        # Create default config
        default_config = {
            'data': {
                'input_path': 'data/raw/',
                'processed_path': 'data/processed/',
                'output_path': 'data/output/'
            },
            'preprocessing': {
                'normalization': 'z-score',
                'fill_na_value': 0
            },
            'split': {
                'train_size': 0.8,
                'test_size': 0.2,
                'random_state': 0,
                'stratify': True
            },
            'feature_selection': {
                'mutual_info_threshold': 0.9,
                'fcbf_k': 20
            },
            'smote': {
                'sampling_strategy': 'auto',
                'random_state': 42
            },
            'models': {
                'xgboost': {
                    'n_estimators': 70,
                    'max_depth': 14,
                    'learning_rate': 0.734
                },
                'random_forest': {
                    'n_estimators': 71,
                    'max_depth': 46,
                    'min_samples_split': 9,
                    'min_samples_leaf': 1,
                    'max_features': 20,
                    'criterion': 'entropy',
                    'random_state': 0
                },
                'decision_tree': {
                    'max_depth': 47,
                    'min_samples_split': 3,
                    'min_samples_leaf': 2,
                    'max_features': 19,
                    'criterion': 'gini',
                    'random_state': 0
                },
                'extra_trees': {
                    'n_estimators': 53,
                    'max_depth': 31,
                    'min_samples_split': 5,
                    'min_samples_leaf': 1,
                    'max_features': 20,
                    'criterion': 'entropy',
                    'random_state': 0
                },
                'stacking_meta': {
                    'n_estimators': 30,
                    'max_depth': 36,
                    'learning_rate': 0.192
                }
            }
        }
        
        # Create config directory
        config_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Save default config
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"  ✓ Created default config at {config_path}")
    
    # Train model
    trainer = ThreatModelTrainer(config_path=args.config)
    accuracy = trainer.train(args.data_path)
    
    # Save model
    success = trainer.save_model(models_dir=args.output)
    
    if success:
        print(f"\n🎉 Training pipeline completed successfully!")
        print(f"   Final accuracy: {accuracy:.4f}")
        print(f"   Models saved to: {args.output}/")
    else:
        print(f"\n⚠ Training completed but model saving failed")
        print(f"   Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
